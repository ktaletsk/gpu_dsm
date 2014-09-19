/*
 * timer.h
 *
 *  Created on: Feb 2, 2012
 *      Author: marat
 */

#ifndef TIMER_H_
#define TIMER_H_
#include <sys/time.h>
//realisation is different for linux/windows
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
