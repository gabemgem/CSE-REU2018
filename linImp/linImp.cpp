// linImp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;


int main()
{
	char SEP = ',';//separator character
	char OPEN = '[';//open delimited section character
	char CLOSE = ']';//close delimited section character
	char ESC = '\\';//escape character
	string FILENAME = "input.txt";

	ifstream input(FILENAME);
	
	if (!input) {
		cout << "Cannot open file.\n";
		return 1;
	}

	string s;
	getline(input, s);
	unsigned int siz = s.size();

	vector<int> escape(siz, 0);//vector with locations of escape characters
	for (unsigned int i = 0; i < s.size(); ++i) {//find locations of escape characters
		if (s[i] == ESC)
			escape[i] = 1;
		else
			escape[i] = 0;
	}

	vector<int> open(siz, 0);//vector with locations of open characters
	vector<int> close(siz, 0);//vector with locations of close characters
	
	for (unsigned int j = 0; j < s.size(); ++j) {//find open and close characters
		if (s[j] == OPEN) {//if found an open
			open[j] = 1;
		}
		if (s[j] == CLOSE) {//if found a close
			close[j] = 1;
		}
	}

	//function for each character:
	//D(i) = (D(i-1) & (!C(i) | E(i-1)) | O(i)

	vector<vector<int> > func(siz);//2D vector to hold the functions to compute delimited sections
	vector<int> col = { 0,0 };
	for (unsigned int i = 0; i < siz; ++i)//initialize vector
		func[i] = col;
	func[0][0] = func[0][1] = open[0];
	for (unsigned int i = 1; i < siz; ++i) {//compute the function for each character
		func[i][0] = open[i];
		func[i][1] = !close[i] | escape[i - 1] | open[i];
	}

	vector<int> delsec(siz, 0);//vector to record delimited sections
	if (s[0] == OPEN)//set initial value
		delsec[0] = 1;
	for (unsigned int i = 1; i < siz; ++i) {//find delimited sections
		delsec[i] = func[i][delsec[i - 1]];
	}
	
	/*for (int p = 0; p < delsec.size(); ++p) {//Delimiter vector printout for debugging
		cout << delcount[p];
	}
	cout << "\n\n";*/

	vector<int> separators(siz, 0);//vector with locations of separator characters
	
	for (unsigned int k = 0; k < s.size(); ++k) {//find locations of separators
		if (s[k] == SEP && !delsec[k]) {//check if separator is in delimited section
			separators[k] = 1;
		}
	}

	vector<int> separatorCount(siz, 0);//vector to count separator characters
	separatorCount[0] = 0;

	for (unsigned int l = 1; l < separators.size(); ++l) {//perform a linear "parallel scan" of the vector
		separatorCount[l] = separatorCount[l - 1] + separators[l - 1];
	}

	vector<int> ret(separatorCount[separatorCount.size()-1], 0);//vector to hold locations of the separators
	
	for (unsigned int m = 0; m < separatorCount.size() - 1; ++m) {//find locations of separators
		if (separatorCount[m] < separatorCount[m + 1]) {
			ret[separatorCount[m]] = m;
		}
	}
	for (unsigned int n = 0; n < ret.size(); ++n) {//print locations
		cout << ret[n];
		cout << "\n";
	}
    return 0;
}

