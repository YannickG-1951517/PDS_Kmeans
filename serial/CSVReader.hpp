#ifndef CSVREADER_H
#define CSVREADER_H
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <sstream>
#include <algorithm>

class CSVReader {
public:
	CSVReader(std::istream &stream, char delimiter = ',', char comment = '#')
		: stream_(stream)
		, delimiter_(delimiter)
		, comment_(comment) {}

	bool read(std::vector<double> &to) {
		std::string line;

		do {
			getline(stream_, line);
		} while (!stream_.eof() && (line.empty() || line[0] == comment_));

		if (stream_.eof())
			return false;

		size_t count = std::count(line.begin(), line.end(), delimiter_) + 1;

		to.resize(count);
		size_t endPos, pos = 0;

		try  {
			for (size_t i = 0; i < count; ++i) {
				endPos = line.find(delimiter_, pos);
				if (std::string::npos != endPos) {
					to[i] = std::stod(line.substr(pos, endPos - pos));
					pos = endPos + 1;
				} else {
					to[i] = std::stod(line.substr(pos));
				}
			}
		} catch (const std::invalid_argument &) {
			std::cerr << "Can't convert '" << line.substr(pos, endPos - pos) << "'" << std::endl;
			return false;
		} catch (const std::out_of_range &) {
			std::cerr << "Argument is out of range for a double" << std::endl;
			return false;
		}

		return true;
	}
private:
	std::istream &stream_;
	const char delimiter_;
	const char comment_;
};

#endif /* CSVREADER_H */
