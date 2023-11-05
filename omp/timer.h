#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

class Timer {
public:
    Timer(bool autoStart = true) {
		if (autoStart)
			start();
    }

    void start() {
        beg = std::chrono::steady_clock::now();
    }

    void stop() {
        end = std::chrono::steady_clock::now();
    }

    double durationNanoSeconds() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count();
    }
private:
    std::chrono::time_point<std::chrono::steady_clock> beg;
    std::chrono::time_point<std::chrono::steady_clock> end;
};

class AutoAverageTimer : public Timer
{
public:
	AutoAverageTimer(const std::string &name) : Timer(false), m_name(name), m_justReported(false) { }
	~AutoAverageTimer()
	{
		if (!m_justReported)
			report();
	}

	void stop()
	{
		Timer::stop();
		m_times.push_back(durationNanoSeconds());
		m_justReported = false;
	}

	void report(std::ostream &o = std::cerr, double multiplier = 1e-9, const std::string suffix = " sec")
	{
		double avg = 0;
		for (auto x : m_times)
			avg += x;
		if (m_times.size() > 0)
			avg /= m_times.size();
		double stddev = 0;
		for (auto x : m_times)
		{
			double dt = x - avg;
			stddev += dt*dt;
		}
		if (m_times.size() > 0)
			stddev /= m_times.size();
		stddev = std::sqrt(stddev);
		o << "#" << m_name << " " << avg*multiplier << " +/- " << stddev*multiplier << suffix << " (" << m_times.size() << " measurements)" << std::endl;
		m_justReported = true;
	}
private:
	std::string m_name;
	std::vector<double> m_times;
	bool m_justReported;
};
