#pragma once

#include <vector>
#include <fstream>

class CSVWriter
{
public:
	CSVWriter(std::ostream &stream, char delimiter = ',')
		: stream_(stream), delimiter_(delimiter) { }

	void write(const std::vector<double> &row, const std::string &linePrefix = "") { write(row.data(), row.size(), linePrefix); }
	template<class X> void write(const std::vector<X> &row, const std::string &linePrefix = "");
	
	// elements in row-major ordering, ie elements of the same row come
	// one after the other
	void write(const std::vector<double> &data, size_t numCols, const std::string &linePrefix = "");
	template<class X> void write(const std::vector<X> &data, size_t numCols, const std::string &linePrefix = "");

	template<class X> void write(const std::vector<std::vector<X>> &rows, const std::string &linePrefix = "");
private:
	void write(const double *row, size_t numCols, const std::string &linePrefix);

	std::ostream &stream_;
	char delimiter_;
};

class FileCSVWriter : public CSVWriter
{
public:
	FileCSVWriter(const std::string &fileName, char delimiter = ',')
		: CSVWriter(internalStream, delimiter)
	{
		open(fileName);
	}

	FileCSVWriter(char delimiter = ',') : CSVWriter(internalStream, delimiter) { }

	void open(const std::string &fileName) { internalStream.open(fileName); }
	bool is_open() const { return internalStream.is_open(); }
	void close() { internalStream.close(); }
private:
	std::ofstream internalStream;
};

inline void CSVWriter::write(const double *row, size_t numCols, const std::string &linePrefix)
{
	if (numCols > 0)
	{
		char buffer[128];
		auto writeNumber = [&buffer,this](double x)
		{
			snprintf(buffer, 128, "%.10g", x);
			stream_ << buffer;
		};

		stream_ << linePrefix;
		writeNumber(row[0]);

		for (size_t i = 1; i < numCols ; i++)
		{
			stream_ << delimiter_;
			writeNumber(row[i]);
		}
	}
	stream_ << std::endl;
}

template<class X> 
inline void CSVWriter::write(const std::vector<X> &row, const std::string &linePrefix)
{
	if (row.size() > 0)
	{
		stream_ << linePrefix;
		stream_ << row[0];
		for (size_t i = 1 ; i < row.size() ; i++)
		{
			stream_ << delimiter_;
			stream_ << row[i];
		}
	}
	stream_ << std::endl;
}

inline void CSVWriter::write(const std::vector<double> &data, size_t numCols, const std::string &linePrefix)
{
	if (numCols == 0)
		throw std::runtime_error("number of columns must be at least one");
	if (data.size()%numCols != 0)
		throw std::runtime_error("data length is not a multiple of specified number of columns");
	
	size_t numRows = data.size()/numCols;
	const double *pRow = data.data();
	for (size_t r = 0 ; r < numRows ; r++, pRow += numCols)
		write(pRow, numCols, linePrefix);
}

template<class X>
inline void CSVWriter::write(const std::vector<std::vector<X>> &rows, const std::string &linePrefix)
{
	for (auto &r : rows)
		write(r, linePrefix);
}

template<class X>
void CSVWriter::write(const std::vector<X> &data, size_t numCols, const std::string &linePrefix)
{
	if (numCols == 0)
		throw std::runtime_error("number of columns must be at least one");
	if (data.size()%numCols != 0)	
		throw std::runtime_error("data length is not a multiple of specified number of columns");

	size_t numRows = data.size()/numCols;
	size_t idx = 0;
	for (size_t r = 0 ; r < numRows ; r++)
	{	
		stream_ << linePrefix;
		stream_ << data[idx++];
		for (size_t c = 1 ; c < numCols ; c++)
		{
			stream_ << delimiter_;
			stream_ << data[idx++];
		}
		stream_ << std::endl;
	}
}

