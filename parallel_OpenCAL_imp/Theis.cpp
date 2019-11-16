#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
using namespace std;


double fatt(double n)
{
  if (n < 0) return -1;

  if (n == 0) return 1;
  else return n*fatt(n-1);
}


int main()
{
    
    double r[] = {127,150,170,200,300};
   // double r = 150;
    double Q = 0.001 ;// portata
    double Sy = 0.1;
    double t = 86400;
    double T = 0.0000125*50;//K*b
    double var = -0.577216;
    // cout << "  pow(r,2) = " << pow(r,2) << endl;
    // cout << "  pow(r,2)*Sy = " << pow(r,2)*Sy << endl;
    // cout << "  4*t*T = " << 4*t*T << endl;

    double u =0;// (pow(r,2)*Sy)/(4*t*T);
    // cout << "   u  = " << u << endl;
    
    

    int max = 40;
    int numberofdays = 14;
    
    for(int ri = 0; ri < 5; ri++){
        FILE * theis;
        string name = "./theis"+to_string(r[ri])+".txt";
        theis =  fopen(name.c_str(), "w");
        fclose(theis);

        theis =  fopen(name.c_str(), "a");

        for(int day = 1; day < numberofdays; day++){
        
            u = (pow(r[ri],2)*Sy)/(4*(t*day)*T);

            double logu = log(u);
            double W = var - logu + u;
            // cout << u << endl;
            //cout << W << endl;

            for(int i = 2; i < max; i++){
                double powu = pow(u,i);
                // cout << "i = " << i << "   powu  = " << powu << endl;
                double denominatore = ((double)i)*fatt(i);
                // cout << "   denominatore  = "        << denominatore << endl;
                double varTmp = powu/(denominatore);
                // cout << "   varTmp  = "              << varTmp << endl;
                
                if(i%2==0){
                    W-=varTmp;
                }else{
                    W+=varTmp;
                }

                //cout << "   W  = "              << W << endl;

            }

            double s;//theis
            s=(Q/(4*M_PI*T))*W; 
        // cout << "W   " <<W << endl;
            //cout <<  Q/(4*M_PI*T) << endl;
            //cout << s << endl;
            fprintf(theis, "%d %f \n", day, s);
            cout << "seconds = "<< t*day << "  "<<s << endl;

        }
         fclose(theis);
    }

    //cout << ris << endl;
   
    return 0;
}