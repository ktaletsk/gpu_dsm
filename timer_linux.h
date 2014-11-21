// Copyright 2014 Marat Andreev
// 
// This file is part of gpu_dsm.
// 
// gpu_dsm is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// at your option) any later version.
// 
// gpu_dsm is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with gpu_dsm.  If not, see <http://www.gnu.org/licenses/>.

#ifndef TIMER_H_
#define TIMER_H_
#include <sys/time.h>
//Realization is different for Linux/Windows
	class ctimer{
	private:
		timeval ts, te;
		bool running;
	public:
		ctimer(){
			running=false;
			gettimeofday(&ts, NULL);
			gettimeofday(&te, NULL);
		};
		void start(){
			running=true;
			gettimeofday(&ts, NULL);
		};
		void stop(){
			running=false;
			gettimeofday(&te, NULL);
		}
	    double elapsedTime(){
	    	double tmp;
	    	if (running){
	    		timeval tr;
	    		gettimeofday(&tr, NULL);
	    		tmp = (tr.tv_sec - ts.tv_sec) * 1000.0;      // sec to ms
	    		tmp += (tr.tv_usec - ts.tv_usec) / 1000.0;   // us to ms

	    	}else{
	    		tmp = (te.tv_sec - ts.tv_sec) * 1000.0;      // sec to ms
	    		tmp += (te.tv_usec - ts.tv_usec) / 1000.0;   // us to ms

	    	}
	    	return tmp;
	    };
	};



#endif /* TIMER_H_ */
