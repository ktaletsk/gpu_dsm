// Copyright 2015 Marat Andreev, Konstantin Taletskiy, Maria Katzarova
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
#include <time.h>
//realisation is different for linux/windows
	class ctimer{
	private:
		clock_t ts, te;
		bool running;
	public:
		ctimer(){
			running=false;
			ts=clock() / (CLOCKS_PER_SEC / 1000);
			te=clock() / (CLOCKS_PER_SEC / 1000);
		};
		void start(){
			running=true;
			ts=clock() / (CLOCKS_PER_SEC / 1000);
		};
		void stop(){
			running=false;
			te=clock() / (CLOCKS_PER_SEC / 1000);
		}
	    double elapsedTime(){//milliseconds
	    	double tmp;
	    	if (running){
	    		clock_t tr;
			tr=clock() / (CLOCKS_PER_SEC / 1000);
	    		tmp = tr -ts;

	    	}else{
	    		tmp = te -ts;
	    	}
	    	return tmp;
	    };
	};



#endif /* TIMER_H_ */
