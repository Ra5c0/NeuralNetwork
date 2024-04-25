/*!
 *  \file neuralNetwork.c
 *  \author GIGON Oscar <gigonoscar@cy-tech.fr>
 *  \version 0.1
 *  \date Ven 03 Mars - 17:06:38 *
 *
 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define teta 0.5
#define eps 0.01

double inputs[48];
double weights[48];

// affiche un tableau
void print(double *tab)
{
    for (int i = 0; i < 48; i++)
    {
        printf("%f ", tab[i]);
    }
}

// random int entre 0 et 1
int rand01()
{
    int rd = rand() & 1;
    return rd;
}

// initialisation des poids
void init_weights()
{
    for (int i = 0; i < 48; i++)
    {
        weights[i] = (2 *((double)rand()) / ((double)(RAND_MAX)) - 1) / 48;
    }
}

// lis les fichiers zero.txt et un.txt
void read_file(int rd, double *tab)
{
    FILE *fptr;
    char c;
    int i = 0;

    if (rd == 0)
    {
        fptr = fopen("zero.txt", "r");
    }
    else
    {
        fptr = fopen("un.txt", "r");
    }

    while ((c = getc(fptr)) != EOF)
    {
        if (c == '*')
        {
            tab[i] = 1;
            i++;
        }
        else if (c == '.')
        {
            tab[i] = 0;
            i++;
        }
    }

    fclose(fptr);
}

// récupère Y (la valeur de la classe) dans les fichiers zero.txt et un.txt
int get_y(int rd, double *tab)
{
    char c;
    FILE *fptr;

    if (rd == 0)
    {
        fptr = fopen("zero.txt", "r");
    }
    else
    {
        fptr = fopen("un.txt", "r");
    }

    while ((c = getc(fptr)) != EOF)
    {
        if (c >= 48 && c <= 57)
        {
            return (c - '0');
        }
    }
    fclose(fptr);
}

// calcul le potentiel de neurone de sortie
double neural_pot(double *inputs, double *weights)
{
    double pot = 0;

    for (int j = 0; j < 48; j++)
    {
        pot += inputs[j] * weights[j];
    }

    return pot - teta;
}

// calcul la reponse du neurone de sortie
int neural_response(double pot_exit)
{
    if (pot_exit > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

// calcul l'erreur
int error_calc(double Y, double X)
{
    double err;
    err = Y - X;
    return err;
}

// apprentissage
void learn(double err, double *inputs)
{
    for (int i = 0; i < 48; i++)
    {
        weights[i] = weights[i] + eps * err * inputs[i];
    }
}

// calcul l'erreur totale
int error_tot(int fileNum)
{
    init_weights();
    int rd = rand01();
    read_file(rd, inputs);
    double Y = get_y(rd, inputs);
    double pot = neural_pot(inputs, weights);
    double X = neural_response(pot);
    int err = error_calc(Y, X);
    return (err);
}

// inverse la valeur des pixels en fonction du pourcentage de bruit souhaité
void invert_pixels(double *pixels, double noise_percent) 
{
    int num_pixels = 48;
    int num_noise_pixels = (int)(num_pixels * noise_percent / 100);
    
    for (int i = 0; i < num_noise_pixels; i++) 
    {
        int index = rand() % num_pixels;
        pixels[index] = 1.0 - pixels[index];
    }
}

// compte le nombre d'erreurs du réseau pour 50 motifs ayant le même niveau de bruit
int count_errors(double **motifs, double noise_percentage) {
    int i, j, nb_errors = 0;
    double input[48];
    for (i = 0; i < 50; i++) {
        
        for (j = 0; j < 48; j++) {
            input[j] = motifs[i][j];
        }
        
        invert_pixels(input, noise_percentage);
        
        if (i < 25) {
            // Motif 0
            if (input[0] < 0.5) {
                nb_errors++;
            }
        } else {
            // Motif 1
            if (input[0] >= 0.5) {
                nb_errors++;
            }
        }
    }
    return nb_errors;
}

// courbe de généralisation pour un motif donné
void generalization_curve(double **motifs, int motif_index) {
    int i, nb_errors;

    FILE *f;
    if (motif_index == 0)
    {
        f = fopen("graphErrBruit0.txt", "w");
    }
    else
    {
        f = fopen("graphErrBruit1.txt", "w");
    }

    printf("Bruit\tTaux d'erreur\n");
    for (i = 0; i <= 100; i += 10) {
        nb_errors = count_errors(motifs, (double)i);
        printf("%d%%\t%.2f%%\n", i, (double)nb_errors * 2.0);
        fprintf(f, "%d%% %.2f%% \n",i, (double)nb_errors * 2.0);
    }
    fclose(f);
}

// génère un tableau à deux dimensions motifs de taille 50x48
double **generate_motifs(double *motif0, double *motif1) 
{
    double **motifs = malloc(sizeof(double*) * 50);
    int i, j;
    for (i = 0; i < 50; i++) {
        motifs[i] = malloc(sizeof(double) * 48);
        for (j = 0; j < 48; j++) {
            if (i < 25)
            {
                motifs[i][j] = motif0[j];
            }
            else
            {
                motifs[i][j] = motif1[j];
            }
            
            
        }
    }
    return motifs;
}

// libère la mémoire allouée pour le tableau des motifs
void free_motifs(double **motifs) 
{
    int i;
    for (i = 0; i < 50; i++) {
        free(motifs[i]);
    }
    free(motifs);
}

// Exercice 1 : Separation de deux classes, regle du perceptron simple
void exo1()
{
    double ERR;
    FILE *f;
    f = fopen("graphErr.txt", "w");
    int i = 0;
    do
    {
        srand((unsigned)time(NULL));
        init_weights();
        int rd = rand01();
        read_file(rd, inputs);
        double Y = get_y(rd, inputs);
        double pot = neural_pot(inputs, weights);
        double X = neural_response(pot);
        int err = error_calc(Y, X);
        learn(err, inputs);
        ERR = abs(error_tot(1)) + abs(error_tot(0));
        //printf("Erreur totale : %f \n\n", ERR);
        
        fprintf(f, "%d %f \n",i, ERR);
        
        i++;
    } while (ERR > 0);
    fclose(f);
    printf("Exo1 --> iteration : %d \n",i);
}

// généralisation
void generalization()
{
    double motif0[48] = {0,0,0,0,0,0,
                        0,1,1,1,1,0,
                        0,1,0,0,1,0,
                        0,1,0,0,1,0,
                        0,1,0,0,1,0,
                        0,1,0,0,1,0,
                        0,1,1,1,1,0,
                        0,0,0,0,0,0};
    double motif1[48] = {0,0,0,0,0,0,
                        0,0,0,1,0,0,
                        0,0,1,1,0,0,
                        0,1,0,1,0,0,
                        0,1,0,1,0,0,
                        0,0,0,1,0,0,
                        0,1,1,1,1,0,
                        0,0,0,0,0,0};

    double **motifs = generate_motifs(motif0,motif1);
    
    printf("Courbe de generalisation pour le motif 0 :\n");
    generalization_curve(motifs, 0);
    printf("\nCourbe de generalisation pour le motif 1 :\n");
    generalization_curve(motifs, 1);
    free_motifs(motifs);

}

// Exercice 2 : Separation de deux classes, regle de Widrow-Hoff
//void exo2(){}

int main()
{
    exo1();
    printf("\n\n");

    generalization();
    printf("\n\n");

    return 0;
}