#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <chrono>

using namespace std;

double sigmoid(double z) {
	return 1.0 / (1.0 + exp(-z));
}

double calculateLoss(const vector<vector<double>>& X, const vector<double>& y, const vector<double>& weights, double bias) {
	double total_loss = 0.0;
	int m = X.size();
	for (int i=0; i<m; i++) {
		double z = bias;
		for (size_t j = 0; j<weights.size(); j++) {
			z += weights[j] * X[i][j];
		}
		double p = sigmoid(z);
		p = max(1e-15, min(1.0 - 1e-15, p));
		total_loss += - (y[i] * log(p) + (1.0 - y[i]) * log(1.0 - p));
	}
	return total_loss / m;
}

int main() {
	cout << "\033[2J\033[1;1H";
	cout << "\033[1;20r";
	
	cout << "=================================================================================\n";
	cout << "                                    TRAIN                                        \n";
	cout << "=================================================================================\n\n";
	
	ifstream file("dataset.csv");
	if (!file.is_open()) {
		cout << "\033[r";
		cerr << "[ERROR] Impossible to open dataset.csv\n";
		return 1;
	}	
	
	string line;
	getline(file, line);
	
	vector<vector<double>> X;
	vector<double> y;
	
	while (getline(file, line)) {
		if (line.empty()) continue;
		stringstream ss(line);
		string val;
		vector<double> row;
		
		for (int i = 0; i < 7; i++) {
			getline(ss, val, ',');
			row.push_back(stod(val));
			}
			X.push_back(row);
			
			getline(ss, val, ',');
			y.push_back(stod(val));
		}
		file.close();
		
		int m = X.size();
		int n = 7;
		cout << "[+] Loaded " << m << " examples from dataset.csv\n";
		
		vector<double> weights(n, 0.0);
		double bias = 0.0;
		
		ifstream inFile("weights.csv");
		if (inFile.is_open()) {
			for (int j = 0; j < n; j++) {
				inFile >> weights[j];
			}
			inFile >> bias;
			inFile.close();
			cout << "[+] Found pre-existent weights in weights.csv\n";
		} else {
			cout << "[!] No pre-existent weight found. Starting frest\n";
		}
		
		double initial_loss = calculateLoss(X, y, weights, bias);
		cout << fixed << setprecision(5);
		cout << "[+] Initial loss: " << initial_loss << "\n\n";
		cout << "Press Enter to begin gradient descent...";
		cin.get();
		
		double learning_rate = 0.01; 
		int epochs = 10000;
		int bar_width = 50;
		
		auto start_time = chrono::high_resolution_clock::now();
		
		for (int epoch = 1; epoch <= epochs; epoch++) {
			vector<double> dw(n, 0.0);
			double db = 0.0;
			
			for (int i = 0; i < m; i++) {
				double z = bias;
				for (int j = 0; j < n; j++) {
					z += weights[j] * X[i][j];
				}
				double prediction = sigmoid(z);
				double error = prediction - y[i];
				
				for (int j = 0; j < n; j++) {
					dw[j] += error * X[i][j];
				}
				db += error;
			}
		
			for (int j = 0; j < n; j++) {
				weights[j] -= learning_rate * (dw[j] / m);
			}
			bias += learning_rate * (db / m);
		
			if (epoch % 2 == 0 || epoch == epochs) {
				double current_loss = calculateLoss(X, y, weights, bias);
			
				cout << "\033[20;1H";
				cout << "[EP " << right << setw(5) << epoch
				 	<< " | Loss: " << setw(7) << current_loss
				 	<< " | dW0: " << setw(8) << dw[0] / m
				 	<< " | W0: " << setw(8) << weights[0]
				 	<< " | W1: " << setw(8) << weights[1]
				 	<< " | Bias: " << setw(8) << bias << "\n";
			}
		
			if (epoch % 10 == 0 || epoch == epochs) {
				float progress = (float)epoch / epochs;
				int pos = bar_width * progress;
			
				cout << "\033[22;1H";
				cout << "---------------------------------------------------------------------------------\n";
				cout << " PROGRESS: [";
				for (int i = 0; i < bar_width; i++) {
					if (i < pos) cout << "#";
					else if (i == pos) cout << ">";
					else cout << ".";
				}
				cout << "] " << right << setw(3) << int(progress * 100.0) << " % (" << epoch << "/" << epochs << ")";
				cout.flush();
			}
		}
	
	cout << "\033[r";
	cout << "\033[25;1H\n";
	
	auto end_time = chrono::high_resolution_clock::now();
	chrono::duration<double> duration = end_time - start_time;
	
	double final_loss = calculateLoss(X, y, weights, bias);
	
	cout << "=================================================================================\n";
	cout << "                              TRAINING RESULTS                                   \n";
	cout << "=================================================================================\n";
	cout << "Time needed		: " << duration.count() << " seconds\n";
	cout << "Initial loss		: " << initial_loss << "\n";
	cout << "Final loss		: " << final_loss << " (Improvement: " << (initial_loss - final_loss) << ")\n";
	cout << "---------------------------------------------------------------------------------\n";
	cout << "Optimized Weights	:\n";
	string labels[] = {"Importance", "Preparation", "Anxiety", "Convenience", "Will", "Usefullness", "Effort"};
	for (int j = 0; j < n; j++)  {
		cout << "  - " << left << setw(15) << labels[j] << ": " << setw(8) << weights[j] << "\n";
	}
	cout << "  - " << left << setw(15) << "Bias" << ": " << setw(8) << bias << "\n";
	cout << "=================================================================================\n";
	
	ofstream outFile("weights.csv");
	if (outFile.is_open()) {
		for (int j = 0; j < n; j++) {
			outFile << weights[j] << "\n";
		}
		outFile << bias << "\n";
		outFile.close();
		cout << "[SUCCESS] New weights saved in weights.csv\n\n";
	}
	
	return 0;
}