#include <cmath>
#include <iostream>
#include <fstream>

int main(void){
    int start = 0, end = 10;
    int a[end-start];
    getSeg(a,&start,&end);
    return 0; 
}


int getSeg(int* a,int* start, int* end){
std::ifstream file("data.txt");
int count = 0;
double value;

while (file >> value && count <= *end) {
    std::cout << value << std::endl;

    count++;
}

}



