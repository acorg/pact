#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double static inline decay(double x){
    return 1/(1+exp(-10*x));
}

static void fill_dist_pairs(double* dist, double* z,
                            int* row_coord_indices, int* col_coord_indices, int n_obs,
                            int nrow_names, int ncol_names, int dims)
{
    int k, d;
    double diff, dist_sq;

    /* compute distances only for observed antigen-serum pairs.
    unobserved pairs are never visited; repeated pairs
    are computed once and cached via the -1 sentinel. */

    for (k = 0; k < nrow_names * ncol_names; k++)
        dist[k] = -1.0;

    for (k = 0; k < n_obs; k++) {
        int i = row_coord_indices[k];
        int j = col_coord_indices[k];
        int lin = i * ncol_names + j;

        if (dist[lin] >= 0.0) continue;

        dist_sq = 0.0;
        for (d = 0; d < dims; d++) {
            diff = z[i*dims + d] - z[nrow_names*dims + j*dims + d];
            dist_sq += diff * diff;
        }
        dist[lin] = sqrt(dist_sq);
    }
}


double _stress_and_jacobian_buffered(
    double* z, double* target_distances, int* distance_types,
    int* row_indices, int* col_indices,
    int* row_coord_indices, int* col_coord_indices,
    int n_obs,
    double* row_av_penalties, double* row_av_means,
    double* col_av_penalties, double* col_av_means,
    double* table_bias_penalties, double* table_bias_means,
    int* off_avidities, int* off_table_biases, int* table_indices,
    double step, int nrows, int ncols, int nrow_names, int ncol_names,
    int dims, int n_tables,
    int row_avidity_on, int col_avidity_on, int table_bias_on,
    double* map_distances, double* grad)
{
    int ra, ca, i_av, j_av, i_coord, j_coord, d, k, t, map_lin;
    int coord_end   = (nrow_names + ncol_names) * dims;
    int row_av_base = coord_end;
    int col_av_base = row_av_base + (row_avidity_on ? nrows : 0);
    int tb_base     = col_av_base + (col_avidity_on ? ncols : 0);
    int total_len   = tb_base     + (table_bias_on  ? n_tables : 0);
    double stress = 0.0;
    double err = 0.0;
    double factor = 0.0;
    double coord1, coord2, decay_val, av_dif;

    t = 0;

    fill_dist_pairs(map_distances, z, row_coord_indices, col_coord_indices,
                    n_obs, nrow_names, ncol_names, dims);

    for (k = 0; k < total_len; k++)
        grad[k] = 0;

    /* row avidity regularization */
    if (row_avidity_on) {
        for (ra = 0; ra < nrows; ra++) {
            av_dif = z[row_av_base + ra] - row_av_means[ra];
            stress += av_dif * av_dif * row_av_penalties[ra];
            if (off_avidities[ra] == 0)
                grad[row_av_base + ra] += 2.0 * av_dif * row_av_penalties[ra];
        }
    }

    /* col avidity regularization */
    if (col_avidity_on) {
        for (ca = 0; ca < ncols; ca++) {
            av_dif = z[col_av_base + ca] - col_av_means[ca];
            stress += av_dif * av_dif * col_av_penalties[ca];
            if (off_avidities[nrows + ca] == 0)
                grad[col_av_base + ca] += 2.0 * av_dif * col_av_penalties[ca];
        }
    }

    /* table bias regularization */
    if (table_bias_on) {
        for (t = 0; t < n_tables; t++) {
            av_dif = z[tb_base + t] - table_bias_means[t];
            stress += av_dif * av_dif * table_bias_penalties[t];
            if (off_table_biases[t] == 0)
                grad[tb_base + t] += 2.0 * av_dif * table_bias_penalties[t];
        }
    }

    /* observation loop */
    for (k = 0; k < n_obs; k++) {

        i_av    = row_indices[k];
        j_av    = col_indices[k];
        i_coord = row_coord_indices[k];
        j_coord = col_coord_indices[k];
        map_lin = i_coord * ncol_names + j_coord;

        if (distance_types[k] == 1 || fabs(map_distances[map_lin]) < 1e-7)
            continue;

        err = target_distances[k] - map_distances[map_lin];
        if (row_avidity_on) err += z[row_av_base + i_av];
        if (col_avidity_on) err += z[col_av_base + j_av];
        if (table_bias_on) { t = table_indices[k]; err += z[tb_base + t]; }

        if (distance_types[k] == 0 || distance_types[k] == 3) {
            stress += err * err;
            factor = 2.0 * err / map_distances[map_lin];
            if (row_avidity_on && off_avidities[i_av] == 0)
                grad[row_av_base + i_av] += 2.0 * err;
            if (col_avidity_on && off_avidities[nrows + j_av] == 0)
                grad[col_av_base + j_av] += 2.0 * err;
            if (table_bias_on && off_table_biases[t] == 0)
                grad[tb_base + t] += 2.0 * err;
        }
        else if (distance_types[k] == 2) {
            err += step;
            decay_val = decay(err);
            stress += err * err * decay_val;
            decay_val *= (1.0 + 5.0 * err * (1.0 - decay_val));
            factor = (2.0 * err / map_distances[map_lin]) * decay_val;
            if (row_avidity_on && off_avidities[i_av] == 0)
                grad[row_av_base + i_av] += 2.0 * err * decay_val;
            if (col_avidity_on && off_avidities[nrows + j_av] == 0)
                grad[col_av_base + j_av] += 2.0 * err * decay_val;
            if (table_bias_on && off_table_biases[t] == 0)
                grad[tb_base + t] += 2.0 * err * decay_val;
        }

        if (dims <= 3) {
            coord1 = z[i_coord*dims];
            coord2 = z[nrow_names*dims + j_coord*dims];
            grad[i_coord*dims]                       += (coord2-coord1)*factor;
            grad[nrow_names*dims + j_coord*dims]     -= (coord2-coord1)*factor;

            if (dims > 1) {
                coord1 = z[i_coord*dims + 1];
                coord2 = z[nrow_names*dims + j_coord*dims + 1];
                grad[i_coord*dims + 1]                   += (coord2-coord1)*factor;
                grad[nrow_names*dims + j_coord*dims + 1] -= (coord2-coord1)*factor;

                if (dims == 3) {
                    coord1 = z[i_coord*dims + 2];
                    coord2 = z[nrow_names*dims + j_coord*dims + 2];
                    grad[i_coord*dims + 2]                   += (coord2-coord1)*factor;
                    grad[nrow_names*dims + j_coord*dims + 2] -= (coord2-coord1)*factor;
                }
            }
        }
        else {
            for (d = 0; d < dims; d++) {
                coord1 = z[i_coord*dims + d];
                coord2 = z[nrow_names*dims + j_coord*dims + d];
                grad[i_coord*dims + d]                   += (coord2-coord1)*factor;
                grad[nrow_names*dims + j_coord*dims + d] -= (coord2-coord1)*factor;
            }
        }
    }

    return stress;
}
