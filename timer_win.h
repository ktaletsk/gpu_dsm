/*
 * timer.h
 *
 *  Created on: Feb 2, 2012
 *      Author: Marat Andreev
 */

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
