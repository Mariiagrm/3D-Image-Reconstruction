
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <omp.h>
#include "matrix_io.h"
#include "strassen_3d.h"

#define CSV_PATH "../../results/tiempos.csv"
#define LOG_PATH "../../results/strassen_paralelo.log"

static const int DEFAULT_SIZES[] = {128, 256, 512, 1024, 2048, 4096, 8192};
static const int DEFAULT_NUM_SIZES = sizeof(DEFAULT_SIZES) / sizeof(DEFAULT_SIZES[0]);

static int csv_is_empty(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 1;
    return st.st_size == 0;
}

static int csv_has_expected_header(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[256];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    fclose(f);
    return strstr(buf, "tiempo_secuencial_s") != NULL &&
           strstr(buf, "tiempo_paralelo_s")   != NULL &&
           strstr(buf, "speedup")             != NULL;
}

static void benchmark_strassen(double *t_sec_out, double *t_par_out, int n) {
    printf("\n....................... BENCHMARK STRASSEN %dx%d .......................\n", n, n);

    Matrix A = crear_aleatoria(n);
    Matrix B = crear_aleatoria(n);

    printf("  Strassen secuencial...\n");
    double t0 = omp_get_wtime();
    Matrix C_seq = strassen_secuencial(A, B);
    *t_sec_out = omp_get_wtime() - t0;
    printf("  Tiempo secuencial : %.4f s\n", *t_sec_out);

    printf("  Strassen paralelo...\n");
    FILE *log = fopen(LOG_PATH, "w");
    Matrix C_par;
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        { C_par = strassen_paralelo(A, B, log); }
    }
    *t_par_out = omp_get_wtime() - t0;
    if (log) fclose(log);
    printf("  Tiempo paralelo   : %.4f s\n", *t_par_out);
    printf("  Speedup           : %.2fx\n", *t_sec_out / *t_par_out);

    if (verificar_resultados(C_seq, C_par))
        printf("  [ OK ] Resultados coinciden.\n");
    else
        printf("  [ X ] Error matematico en Strassen.\n");

    liberar_matriz(A); liberar_matriz(B);
    liberar_matriz(C_seq); liberar_matriz(C_par);
}

int main(int argc, char *argv[]) {
    const int *sizes = DEFAULT_SIZES;
    int num_sizes = DEFAULT_NUM_SIZES;

    int custom_sizes[16];
    int num_custom = 0;

    if (argc >= 2) {
        for (int i = 1; i < argc && num_custom < 16; ++i) {
            int v = atoi(argv[i]);
            if (v > 0) custom_sizes[num_custom++] = v;
        }
        if (num_custom > 0) {
            sizes = custom_sizes;
            num_sizes = num_custom;
        }
    }

    int need_header = csv_is_empty(CSV_PATH) || !csv_has_expected_header(CSV_PATH);
    FILE *csv = fopen(CSV_PATH, "a");
    if (!csv) {
        fprintf(stderr, "No se pudo abrir %s\n", CSV_PATH);
        return 1;
    }
    if (need_header) {
        fprintf(csv, "n,tiempo_secuencial_s,tiempo_paralelo_s,speedup\n");
    }

    printf("Barrido: %d tamanos. Hilos OMP: %d\n", num_sizes, omp_get_max_threads());

    for (int s = 0; s < num_sizes; ++s) {
        int n = sizes[s];
        double t_sec = 0.0, t_par = 0.0;
        benchmark_strassen(&t_sec, &t_par, n);

        double speedup = (t_par > 0.0) ? (t_sec / t_par) : 0.0;
        fprintf(csv, "%d,%.6f,%.6f,%.4f\n", n, t_sec, t_par, speedup);
        fflush(csv);
    }

    fclose(csv);
    printf("\nTiempos exportados a %s\n", CSV_PATH);
    return 0;
}
