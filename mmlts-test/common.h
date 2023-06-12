#ifndef _COMMON_H
#define _COMMON_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>

#define MAX_MEM (8 * 1024 * 1024)

#define check_return(status, target, ...)				\
	do													\
	{													\
		if (status == target)							\
			break;										\
		std::fprintf(stderr, __VA_ARGS__);				\
		return status;									\
	} while (0)

#define check_othret(status, target, succ, fail)		\
	return status == target ? succ : fail

/**
 * @brief Modified from\n
	https://stackoverflow.com/questions/14539867/how-to-display-a-progress-indicator-in-pure-c-c-cout-printf
 * 
 */
class Progress
{
private:
	float progress_unit;
	float cur_progress;
	int barWidth;

	inline void print_bar()
	{
		std::cerr << "\r[";
		for (int i = 0; i < barWidth; i++) {
			int pos = (int)cur_progress;
			if (i < pos)
				std::cerr << "=";
			else if (i == pos)
				std::cerr << ">";
			else
				std::cerr << " ";
		}
		float percent = cur_progress / (float)barWidth * 100.f;
		percent = percent > 100.f ? 100.f : percent;
		std::cerr << "] " << percent <<"%";
		std::cerr.flush();
	}
public:
	Progress(int width = 100) : barWidth(width), progress_unit(0.f), cur_progress(0.f)
	{
		std::cerr << std::fixed << std::setprecision(2);
	}

	inline void start(int iter, std::string msg)
	{
		std::cerr << msg << std::endl;
		progress_unit = (float)barWidth / (float)iter;
		cur_progress = 0.f;
		print_bar();
	}

	inline void inc(int n = 1)
	{
		cur_progress += progress_unit * (float)n;
		print_bar();
	}

	inline void end()
	{
		inc();
		cur_progress = 0.f;
		std::cerr << "\n";
	}
};

#endif /* _COMMON_H */