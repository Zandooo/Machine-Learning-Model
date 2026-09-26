#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>

using namespace std;

struct Activity {
	string name;
	double importance;
	double preparation;
	double anxiety;
	double convenience;
	double will;
	double usefullness;
	double effort;
};

double sigmoid(double z) {
	return 1.0 / (1.0 + exp(-z));
}

double calculateScore(const Activity& act, const vector<double>& weights, double bias) {
	double z = bias;
	z += weights[0] * act.importance;
	z += weights[1] * act.preparation;
	z += weights[2] * act.anxiety;
	z += weights[3] * act.convenience;
	z += weights[4] * act.will;
	z += weights[5] * act.usefullness;
	z += weights[6] * act.effort;
	
	return sigmoid(z);
}

Activity inputActivity(const string& label) {
	Activity act;
	cout << "\n--- Input datas for: " << label << " ---\n";
	cout << "Name of activity (write 'exit' to quit): ";
	cin >> act.name;
	
	if (act.name == "exit" || act.name == "EXIT") return act;
	
	cout << "Importance (1-10): "; cin >> act.importance;
	cout << "Preparation (1-10): "; cin >> act.preparation;
	cout << "Anxiety (1-10): "; cin >> act.anxiety;
	cout << "Convenience (1-10): "; cin >> act.convenience;
	cout << "Will (1-10): "; cin >> act.will;
	cout << "Usefullness (1-10): "; cin >> act.usefullness;
	cout << "Effort (1-10): "; cin >> act.effort;
	return act;
}

int main() {
	ifstream inFile("weights.csv");
	if(!inFile.is_open()) {
		cerr << "[ERROR] File weights.csv not found. Run train.cpp first\n";
		return 1;
	}
	
	vector<double> weights(7);
	double bias;
	
	for (int i = 0; i < 7; i++) {
		if (!(inFile >> weights[i])) {
			cerr << "[ERROR] Wrong format in file weights.csv\n";
			return 1;
		}
	}
	inFile >> bias;
	inFile.close();
	
	cout << "====================================================\n";
	cout << "                  CHOICE SYSTEM                     \n";
	cout << "====================================================\n";
	
	while(true) {
		Activity act1 = inputActivity("Activity 1");
		if (act1.name == "exit" || act1.name == "EXIT") break;
		
		Activity act2 = inputActivity("Activity 2");
		if (act2.name == "exit" || act2.name == "EXIT") break;
		
		double score1 = calculateScore(act1, weights, bias);
		double score2 = calculateScore(act2, weights, bias);
		
		cout << "\n------------------ RESULTS ------------------\n";
		cout << fixed << setprecision(2);
		cout << act1.name << " -> Choice probability: " << score1 * 100 << "%\n";
		cout << act2.name << " -> Choice probability: " << score2 * 100 << "%\n";
		cout << "-----------------------------------------------\n";
		
		if (abs(score1 - score2) < 0.01) {
			cout << " => The two options are the same for the model.\n";
		} else if (score1 > score2) {
			cout << " => Recommended choice: " << act1.name << "\n";
		} else {
			cout << " => Recommended choice: " << act2.name << "\n";
		}
		cout << "===============================================\n\n";
	}
	
	cout << "\nClosing Program\n";
	return 0;
}