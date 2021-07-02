#include <iostream>
#include "utilites.h"
#include "layer.h"
using namespace std;

int main() {
	vector<vector<float>> training_set_inputs{ { 1,0,0}, {0,1,0},{0,0,1},{0,0,0} };
	vector<vector<float>> training_set_outputs{ { 1,0,0}, {0,1,0},{0,0,1},{0,0,0} };
	//vector<vector<float>> training_set_inputs{ {1,1,1,1} };
	//vector<vector<float>> training_set_outputs{ {1,1,1,1} };

	random randint;

	int inputs = 3;
	int epoches;
	cout << "Epoches: ";
	cin >> epoches;

	network model = network(inputs);
	model.add(16, "sigmoid");
	//model.add(32, "softmax");
	//model.add(16, "relu");
	model.add(4, "sigmoid");
	model.complite();

	model.train(training_set_inputs, training_set_outputs,15, epoches,false);//input,output,dropout,epoches
	cout << "Ready" << endl;
	while (true) {
		vector<float> input;
		for (int i = 1; i <= inputs; i++) {
			int a;
			if (i == 1) {
				cout << endl << "Enter 1st: ";
				cin >> a;
				input.push_back(a);
			}
			else if (i == 2) {
				cout << "Enter 2nd: ";
				cin >> a;
				input.push_back(a);
			}
			else if (i == 3) {
				cout << "Enter 3rd: ";
				cin >> a;
				input.push_back(a);
			}
			else {
				cout << "Enter " << i << "th: ";
				cin >> a;
				input.push_back(a);
			}
		}
		model.predict(input);
	}
	return 0;
}