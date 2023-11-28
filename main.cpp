#include <stdio.h>

double add(double a, double b) {
    return a + b;
}

int main(int argc, char* argv[]) {
    double x = 0.0000001;
    double y = 0.5;
    printf("X: %g\n", x);
    printf("Y: %g\n", y);
    printf("Y+X: %g\n", x+y);
    printf("Y-X: %g\n", x-y);
    double z = add(x,y);
    double a = add(y,-1*x);
    printf("Z: %g\n", z);
    printf("A: %g\n", a);
    printf("(Y+X)-(Y-X): %g\n", (x+y)-(y-x));
    printf("Z-A: %g\n", z-a);
}