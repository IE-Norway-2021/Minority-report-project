/**
 * @file main.cpp
 * @author David González León, Jade Gröli
 * @brief Code used for the building of the hand signs dataset
 * @version 0.1
 * @date 19-11-2021
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "PhotoTaker.h"
#include <iostream>
#include <string>

using namespace std;

int main() {
   cout << "Welcome to the photo taker App." << endl;
   string input;
   PhotoTaker photoTaker;
   while (1) {
      cout << "Please write set if you want to take a set of captures, a digit for a single photo or exit if you want to leave : ";
      cin >> input;
      if (!input.compare("exit")) {
         break;
      } else if (input.length() == 1 && isdigit(input.at(0))) {
         photoTaker.takePicture(input.at(0) - 48);
      } else if (!input.compare("set")) {
         cout << "Press enter to take the picture\n";
         cin.get();
         for (int i = 0; i < 10; ++i) {
            cout << "Photo num " << i;
            cin.get();
            photoTaker.takePicture(i);
         }
         cout << "Set complete\n";
      } else {
         cout << "Wrong input, please try again\n";
      }
   }
   cout << "Closing the app...\n";
}
