#include <cmath>
#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>
using namespace std;

class Law
{
private:
	int inputs;
	vector<float> sum;
	vector<float> output;
	vector<vector<float>> W;
	vector<float> delta;
	int neurons;
	bool input_law;
	vector<vector<float>> related_forward;
	vector<float> derivatives_forward;
	int neurons_forward;
	int law_name;
	float learning_rate;
	vector<string> activation;
	vector<string> killed_activation;
	vector<vector<float>> killed_W;
	string classic_activation;

	float sigmoid(float x) {
		return 1 / (1 + exp(-x));
	}
	float relu(float x) {
		float zero = 0;
		return max(zero, x);
	}
	float calc_forward_derivatives(int neuron) {
		float sum_derivatives = 0;
		for (int i = 0; i < int(neurons_forward); i++) {
			sum_derivatives += related_forward[i][neuron] * derivatives_forward[i];
		}
		return sum_derivatives;
	}
public:
	bool hiden_law;
	bool output_law;
	Law(int input, int neurons_local, bool is_out, bool is_in, bool is_h, vector<string> activation_local, int neurons_forward_local, int name, float l_r) {
		activation = activation_local;
		inputs = input;
		neurons = neurons_local;
		output_law = is_out;
		neurons_forward = neurons_forward_local;
		law_name = name;
		generate_weights(false);
		learning_rate = l_r;
		input_law = is_in;
		hiden_law = is_h;
	}
	void set_forward(int neurons_forward_local) {
		neurons_forward = neurons_forward_local;
	}
	vector<vector<float>> get_W() {
		return W;
	}
	int get_neurons() {
		return neurons;
	}
	int get_inputs() {
		return inputs;
	}
	vector<float> get_output() {
		return output;
	}
	void forward(vector<float> derivatives, vector<vector<float>> w_local, int neuron) {
		neurons_forward = neuron;
		related_forward = w_local;
		derivatives_forward = derivatives;
	}
	void generate_weights(bool regenerating)
	{
		if (regenerating == true) {
			W.clear();
		}
		vector<float> local_W;
		for (int l = 0; l < neurons; l++) {
			if (regenerating == false) {
			}
			for (int i = 0; i < inputs; i++)
			{
				int randNum = rand() % 2;
				if (randNum == 1) {
					local_W.push_back(-1 * (double(rand()) / (double(RAND_MAX) + 1.0)));
				}
				else {
					local_W.push_back(double(rand()) / (double(RAND_MAX) + 1.0));
				}
			}
			W.push_back(local_W);
			local_W.clear();
		}
	}
	void correct_weights()
	{
		srand(time(NULL));
		for (int l = 0; l < neurons; l++) {
			for (int i = 0; i < inputs; i++)
			{
				int randNum = rand() % 2;
				if (randNum == 1) {
					W[l][i] -= W[l][i] * 0.01;
				}
				else {
					W[l][i] += W[l][i] * 0.01;
				}
			}
		}
	}
	vector<float> get_delta(vector<float> error) {
		vector<float> derivatives;
		if (output_law == true) {
			for (int i = 0; i < neurons; i++) {
				if (activation[i] == "sigmoid") {
					derivatives.push_back(output[i] * (1 - output[i]) * error[i]);
				}
				else if (activation[i] == "relu") {
					if (output[i] < 0) {
						derivatives.push_back(0.1);
					}
					else {
						derivatives.push_back(1);
					}
				}
				else {
					derivatives.push_back(0);
				}
			}
		}
		else {
			for (int i = 0; i < neurons; i++) {
				if (activation[i] == "sigmoid") {
					derivatives.push_back((exp(sum[i]) / pow((1 + exp(sum[i])), 2)) * calc_forward_derivatives(i));
				}
				else if (activation[i] == "relu") {
					if (output[i] < 0) {
						derivatives.push_back(0.1);
					}
					else {
						derivatives.push_back(calc_forward_derivatives(i));
					}
				}
				else {
					derivatives.push_back(0);
				}
			}
		}
		delta = derivatives;
		return delta;
	}
	vector<float> think(vector<float> input, int kill_neurons, bool test) {
		if (test == false) {
			srand(time(NULL));
			killed_activation = activation;
			killed_W = W;
			for (int i = 0; i < neurons; i++) {
				int r = rand() % neurons;
				if (rand() % 101 <= kill_neurons) {
					int r = rand() % neurons;
					if (activation[r] == "killed") {
						int r = rand() % neurons;
					}
					activation[r] = "killed";
					for (int l = 0; l < W[r].size(); l++) {
						W[r][l] = 0;
					}
				}
			}
		}
		output.clear();
		sum.clear();
		for (int l = 0; l < neurons; l++) {
			float res = 0;
			for (int i = 0; i < inputs; i++)
			{
				res += input[i] * W[l][i];
			}
			sum.push_back(res);
			if (activation[l] == "sigmoid") {
				output.push_back(sigmoid(res));
			}
			else if (activation[l] == "relu") {
				output.push_back(relu(res));
			}
			else if (activation[l] == "killed") {
				output.push_back(0);
			}
		}
		return output;
	}
	void train(vector<float> input, vector<float> error) {
		W = killed_W;
		for (int l = 0; l < neurons; l++) {
			if (output_law == true) {
				for (int i = 0; i < inputs; i++)
				{
					W[l][i] += input[i] * error[l];
				}
			}
			else if (input_law == true) {
				for (int i = 0; i < inputs; i++)
				{
					if (activation[l] == "sigmoid") {
						W[l][i] += sigmoid(input[i]) * output[l] * (1 - output[l]) * calc_forward_derivatives(l);
					}
					else if (activation[l] == "relu") {
						int derivative;
						if (sum[l] <= 0) {
							derivative = 0.1;
						}
						else {
							derivative = 1;
						}
						W[l][i] += relu(input[i]) * derivative * calc_forward_derivatives(l);
					}
				}
			}
			else if (hiden_law == true) {
				for (int i = 0; i < inputs; i++)
				{
					if (activation[l] == "sigmoid") {
						W[l][i] += input[i] * output[l] * (1 - output[l]) * calc_forward_derivatives(l);
					}
					else if (activation[l] == "relu") {
						int derivative;
						if (sum[l] <= 0) {
							derivative = 0.1;
						}
						else {
							derivative = 1;
						}
						W[l][i] += input[i] * derivative * calc_forward_derivatives(l);
					}
				}
			}
			else {

			}
		}
		activation = killed_activation;
	}
};

class network {
private:
	vector<Law> layes;
	int count_layes;
	int inputs;
	float former_val_W = 0.1;
	vector<vector<float>> error;
	vector<float> output;
	float former_error = 0;
public:
	network(int input_local) {
		inputs = input_local;
		count_layes = 0;
	}
	void add(int neurons, string local_activation) {
		count_layes += 1;
		vector<string> activation;
		for (int i = 0; i < neurons; i++) {
			activation.push_back(local_activation);
		}
		if (count_layes == 1) {
			layes.push_back(Law(inputs, neurons, false, true, false, activation, 0, count_layes, 1));
		}
		else {
			layes[count_layes - 2].set_forward(neurons);
			layes.push_back(Law(layes[count_layes - 2].get_neurons(), neurons, false, false, true, activation, 0, count_layes, 1));
		}
	}
	void add_extended(int neurons, vector<string> activation) {
		count_layes += 1;
		if (count_layes == 1) {
			layes.push_back(Law(inputs, neurons, false, true, false, activation, 0, count_layes, 1));
		}
		else {
			layes[count_layes - 2].set_forward(neurons);
			layes.push_back(Law(layes[count_layes - 2].get_neurons(), neurons, false, false, true, activation, 0, count_layes, 1));
		}
	}
	void complite() {
		layes[count_layes - 1].output_law = true;
		layes[count_layes - 1].hiden_law = false;
	}
	void predict(vector<float> input) {
		for (int i = 0; i < layes[count_layes - 1].get_neurons(); i++) {
			cout << think(input, layes, 0, true,false)[i] << endl;
		}
	}
	void generating_weight() {
		for (int i = 0; i < count_layes; i++) {
			layes[i].generate_weights(true);
		}
	}
	vector<float> find_error_with_input(vector<float> inputs, vector<float>answears, vector<Law> testing_layes, int count_output,bool save_layes) {
		vector<float> c_error;
		vector<float> outputing;
		for (int l = 0; l < inputs.size(); l++) {
			outputing = think(inputs, testing_layes, 0, false, save_layes);
		}
		for (int k = 0; k < count_output; k++) {
			if (-(answears[k] - outputing[k]) > (answears[k] - outputing[k])) {
				c_error.push_back(-(answears[k] - outputing[k]));
			}
			else {
				c_error.push_back((answears[k] - outputing[k]));
			}
		}
		return c_error;
	}
	float find_error_with_error(vector<vector<float>> error_local, int count_output) {
		float c_error = 0;
		for (int k = 0; k < error_local.size(); k++) {
			for (int l = 0; l < count_output; l++) {
				c_error += error_local[k][l];
			}
		}
		return c_error;
	}
	float complexy() {
		float val_W = 0;
		for (int j = 0; j < count_layes; j++) {
			float all_w = 0;
			vector<vector<float>> W;
			W = layes[j].get_W();
			for (int p = 0; p < W.size(); p++) {
				for (int v = 0; v < W[p].size(); v++) {
					all_w += W[p][v];
				}
			}
			val_W += all_w / (W.size() * W[0].size());
		}
		if (-val_W > val_W) {
			val_W = -val_W;
		}
		former_val_W = val_W;
		return val_W;
	}
	vector<float> think(vector<float> inputs, vector<Law>testing_layes, int dropout, bool test,bool save_layes) {
		vector<float> res;
		vector<float> old_res;
		for (int i = 0; i < testing_layes.size(); i++) {
			if (i == 0) {
				res = testing_layes[i].think(inputs, dropout, test);
			}
			else if (i == count_layes - 1) {
				old_res = res;
				res = testing_layes[i].think(old_res, 0, test);
			}
			else {
				old_res = res;
				res = testing_layes[i].think(old_res, dropout, test);
			}
		}
		if (save_layes == true) {
			layes = testing_layes;
		}
		return res;
	}
	void train(vector<vector<float>> input_local, vector<vector<float>> answears, int dropout, int epoches, bool gen) {
		former_val_W = complexy();
		float val_error = 0;
		if (gen == false) {
			for (int i = 1; i <= epoches; i++) {
				for (int l = 0; l < answears.size(); l++) {
					error.push_back(find_error_with_input(input_local[l], answears[l], layes, layes[count_layes-1].get_neurons(),true));
				}

				vector<vector<vector<float>>> error_local;
				error_local.push_back(error);
				val_error = classic_train(input_local, error_local,dropout,i,gen);

				error_local.clear();
				error.clear();

				if (val_error == 0) {
					dropout += 5;
					cout << "Increased dropout" << endl;
				}
				else if (val_error > 5 || val_error < 0.001 && dropout != 0) {
					dropout -= 5;
					cout << "Reducing dropout" << endl;
				}
			}
		}
		else {
			for (int i = 1; i <= epoches; i++) {
				cout << endl << "Epoch: " << i << endl;
				int generations = 100;
				vector<vector<vector<float>>> error_local;
				vector<vector<float>> error_local_local;
				for (int l = 0; l < answears.size(); l++) {
					error_local_local.push_back(find_error_with_input(input_local[l],answears[l],layes, layes[count_layes - 1].get_neurons(),false));
				}
				error_local.push_back(error_local_local);
				error_local_local.clear();
				vector<vector<Law>>gen_layes;

				for (int p = 0; p < generations; p++) {
					gen_layes.push_back(layes);
				}//Создание копий

				for (int p = 0; p < generations; p++) {
					for (int k = 0; k < count_layes; k++) {
						gen_layes[p][k].correct_weights();
					}
				}//Создание мутаций в копиях
				for (int k = 0; k < generations; k++) {
					for (int l = 0; l < answears.size(); l++) {
						error_local_local.push_back(find_error_with_input(input_local[l], answears[l], gen_layes[k], layes[count_layes - 1].get_neurons(),false));
					}
					error_local.push_back(error_local_local);
					error_local_local.clear();
				}//Тестирование копий и нахождение ошибки у них
				gen_train(gen_layes, error_local, i);
			}
		}
	}
	float classic_train(vector<vector<float>> input_local, vector<vector<vector<float>>> error_local, int dropout, int epoch, bool gen) {
		cout << endl << "Epoch: " << epoch << endl;
		float val_error = 0;
		for (int l = 0; l < input_local.size(); l++) {
			error = error_local[0];

			for (int k = count_layes - 2; k >= 0; k--) {
				layes[k].forward(layes[k + 1].get_delta(error[l]), layes[k + 1].get_W(), layes[k + 1].get_neurons());
			}

			for (int k = count_layes - 1; k >= 0; k--) {
				if (k == 0) {
					layes[k].train(input_local[l], error[l]);
				}
				else {
					layes[k].train(layes[k - 1].get_output(), error[l]);
				}
			}
			error.clear();
		}
		if (epoch % 100 == 0) {
			float val_W = complexy();
			cout << endl << "Computational complexity on epoch " << epoch << ": " << val_W << ". That more than former in : " << ((val_W / former_val_W) - 1) * 100 << "%" << endl;
		}
		cout << "Error: " << val_error / input_local.size() << endl;
		return val_error;
	}
	void gen_train(vector<vector<Law>>input_gen, vector<vector<vector<float>>>error_local,int epoch) {
		cout << endl << "Epoch: " << epoch << endl;
		int generations = 100;

		vector<float> val_error;
		vector<vector<Law>>gen_layes = input_gen;

		for (int k = 1; k < (generations + 1); k++) {
			val_error.push_back(find_error_with_error(error_local[k], layes[count_layes - 1].get_neurons()));
		}//Нахождение у них ошибки

		float min_error;
		if (-val_error[0] > val_error[0]) {
			min_error = -val_error[0];
		}
		else {
			min_error = val_error[0];
		}//За минимальную ошибку по умолчанию берётся первое значение в векторе

		int index = 0;
		for (int p = 1; p < generations; p++) {
			if (min_error > val_error[p]) {
				min_error = val_error[p];
				index = p;
			}
		}//Нахождение лучшей копии

		float current_error = find_error_with_error(error_local[0], layes[count_layes - 1].get_neurons());//Тестирование основной модели и нахождение у неё ошибки

		if (min_error < current_error) {
			layes = gen_layes[index];
			cout << "Good generation. Min error: " << min_error << " current error: " << current_error << endl;
		}
		else if (min_error > current_error) {
			cout << "Bad generation. Min error: " << min_error << " current error: " << current_error << endl;
			min_error = current_error;
		}

		cout << "Total error: " << min_error << endl;

		if (epoch % 100 == 0) {
			float val_W = complexy();
			cout << endl << "Computational complexity on epoch " << epoch << ": " << val_W << ". That more than former in : " << ((val_W / former_val_W) - 1) * 100 << "%" << endl;
		}
		val_error.clear();
		gen_layes.clear();
	}
};