#include <iostream>
#include <cmath>
#include "layer.h"
#include <thread>
using namespace std;

//class randomization {
//private:
//	int a = 1103515245;
//	int c = 12345;
//	int m = 32768;
//	int next = time(NULL);
//public:
//	int rand()
//	{
//		next = fabs((a * next) + c);
//		return ((next / 65536) % m);
//	}
//	void srand(int seed)
//	{
//		next = seed;
//	}
//};

int main() {
	vector<vector<float>> training_set_inputs{ {1.2,1.2,1.2,1.2}, {1.2,1.2,0.8,1.2},{0.5,1.5,1,1.5},{0.5,0.5,2,0.5} };
	vector<vector<float>> training_set_outputs{ {1,0}, {1,0},{0,1},{0,1} };//buy,sell

	int inputs = 4;
	int epoches;
	cout << "Epoches: ";
	cin >> epoches;

	network model = network(inputs);
	//model.add(32, "sigmoid");
	/*model.add(16, "relu");
	model.add(2, "sigmoid");*/

	model.add(128, "sigmoid");
	model.add(64, "sigmoid");
	model.add(32, "sigmoid");
	model.add(16, "sigmoid");
	model.add(2, "sigmoid");
	model.complite();

	model.classic_train(training_set_inputs, training_set_outputs,0, epoches);//input,output,dropout,epoches
	cout << "Ready" << endl;
	while (true) {
		vector<float> input;
		for (int i = 1; i <= inputs; i++) {
			double a;
			if (i == 1) {
				cout << endl << "Enter 1st: ";
				cin >> a;
				input.push_back(float(a));
			}
			else if (i == 2) {
				cout << "Enter 2nd: ";
				cin >> a;
				input.push_back(float(a));
			}
			else if (i == 3) {
				cout << "Enter 3rd: ";
				cin >> a;
				input.push_back(float(a));
			}
			else {
				cout << "Enter " << i << "th: ";
				cin >> a;
				input.push_back(float(a));
			}
		}
		model.predict(input);
	}
	return 0;
}