#include <iostream>
#include <string>
#include "PhotoTaker.h"

using namespace std;

int main() {
    cout << "Welcome to the photo taker App." << endl;
    string input;
    PhotoTaker photoTaker;
    while (1) {
        cout << "Please write the digit you want to capture (0-9) or exit if you want to leave : ";
        cin >> input;
        if (!input.compare("exit")) {
            break;
        } else if (input.length() == 1 && isdigit(input.at(0))) {
            photoTaker.takePicture(input.at(0) - 48);
        } else {
            cout << "Wrong input, please try again\n";
        }
    }
    cout<<"Closing the app...\n";
}
