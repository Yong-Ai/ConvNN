#include "stdafx.h"
#include "image.h"
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TARGET_SIZE 10000	
#define CARTEGORY 10
#define NUM_EPOCH 100	
#define numberOfWeightFile 1
Img **arch;
char input[40] = ".\\testData\\1.bmp";
int layers[7] = {1,8,8,24,24,100,10};
int filter_layers[6] = { 16, 8, 192, 24, 2400, 500};
int filter_size[6] = {5,4,6,3,6,1};
Weight ***weights;

int target[TARGET_SIZE][CARTEGORY];
int resultTarget[TARGET_SIZE];
double learning_gain = 0.02;

void Img_alloc(Img* arch);
void Convolution(int i, int j);
void Subsampling(int i, int j);
void MLP(int i, int j);
void Cal_delta(Img *arch_before, Img *arch_after, Weight *weight, int mode);
void Update_weight(Img *arch_before, Img *arch_after, Weight *weight, int mode);
void init_delta();
void Save_Weight();
void Save_Weight(int epoch, double energy);
void LoadWeight();
void LoadWeight(int number);
IplImage *src;		

int main()
{		
	FILE *fp_target,*fp;
	int temp_target;
	int count_for_filter_layer = 0;
	int epoch = NUM_EPOCH;
	double energy = 0;
	double temp = -1;
	int Label = -1;	
	
	fp_target = fopen("testData\\target.txt", "r");
	for(int i = 0; i < TARGET_SIZE; i++){
		fscanf(fp_target, "%d", &temp_target);
		for(int j = 0; j < CARTEGORY; j++){
			resultTarget[i] = temp_target;
			if(j == temp_target){
				target[i][j] = 1;
			}
			else{
				target[i][j] = -1;
			}
		}
	}

	src = cvLoadImage(input, 0);

	arch = (Img**)calloc(7, sizeof(Img*));	
	for(int i = 0; i < 7; i++){
		arch[i] = (Img*)calloc(layers[i], sizeof(Img));		
	}

	for(int i = 0; i < layers[0]; i++){
		arch[0][i].height = src->height;
		arch[0][i].width = src->width;
		Img_alloc(&arch[0][i]);
	
	}

	for(int i = 1; i < 7; i++){
		if(i % 2 == 0){
			for(int j = 0; j < layers[i]; j++){
				arch[i][j].height = arch[i-1][0].height / filter_size[count_for_filter_layer];
				arch[i][j].width = arch[i-1][0].width / filter_size[count_for_filter_layer];				
				Img_alloc(&arch[i][j]);
			}
			count_for_filter_layer++;
		}
		else{
			for(int j = 0; j < layers[i]; j++){
				arch[i][j].height = arch[i-1][0].height-filter_size[count_for_filter_layer]+1;
				arch[i][j].width = arch[i-1][0].width-filter_size[count_for_filter_layer]+1;				
				Img_alloc(&arch[i][j]);
			}
			count_for_filter_layer++;
		}
	}



	weights = (Weight***)calloc(6, sizeof(Weight**));
	for(int i = 0; i < 5; i++){
		if(i % 2 == 0){
			weights[i] = (Weight**)calloc(layers[i+1], sizeof(Weight*));			
			for(int j = 0; j < layers[i+1]; j++){
				weights[i][j] = (Weight*)calloc(layers[i], sizeof(Weight));
			}
		}else{
			weights[i] = (Weight**)calloc(layers[i+1], sizeof(Weight*));
			for(int j = 0; j < layers[i+1]; j++){
				weights[i][j] = (Weight*)calloc(1, sizeof(Weight));
			}
		}
	}
	
	weights[5] = (Weight**)calloc(layers[6], sizeof(Weight*));
	for(int i = 0; i < layers[6]; i++){
		weights[5][i] = (Weight*)calloc(layers[5], sizeof(Weight));		
	}

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.25, 0.25);

	for(int i = 0; i < 5; i++){
		if(i % 2 == 0){
			for(int j = 0; j < layers[i+1]; j++){
				for(int k = 0; k < layers[i]; k++){
					weights[i][j][k].filter = (double**)calloc(filter_size[i], sizeof(double*));
					for(int l = 0; l < filter_size[i]; l++){
						weights[i][j][k].filter[l] = (double*)calloc(filter_size[i], sizeof(double));
						for(int m = 0; m < filter_size[i]; m++){
							weights[i][j][k].filter[l][m] =  ((rand() % 201) * 0.01) -1;
							weights[i][j][k].height = filter_size[i];
							weights[i][j][k].width = filter_size[i];							
						}
					}
				}
			}

		}
		else{
			for(int j = 0; j < layers[i+1]; j++){
				weights[i][j][0].beta = ((rand() % 201) * 0.01) -1;
				weights[i][j][0].height = filter_size[i];
				weights[i][j][0].width = filter_size[i];			
			}
		}
	}

	for(int i = 0; i < layers[6]; i++){
		for(int j = 0; j < layers[5]; j++){
			weights[5][i][j].beta = ((rand() % 201) * 0.01) -1;
			weights[5][i][j].height = 1;
			weights[5][i][j].width = 1;
		}		
	}
	cvReleaseImage(&src);
	float tempCount;
	char Name[50];
	char OutName[50];
	char buffer[25];
	//for(int p =1; p<=numberOfWeightFile; p++)
	int p = 7;
	{
		int trueCount = 0;
		LoadWeight(p);
		sprintf(Name, "loadWeight\\result-%d.txt", p);
		fp = fopen(Name,"w");
		for(int it = 0; it < TARGET_SIZE; it++){			
			for(int i = 0; i < 7; i++){
				if(i == 0){
					for(int j = 0; j < layers[i]; j++){
						sprintf(input, ".\\testData\\%d.bmp", it+1);
						fprintf(fp, "%dst\t", it+1);
						src = cvLoadImage(input, 0);
						Make_first(src,&arch[0][j]);
						//Test_showimage(arch[0][j], output_index++);
					}
				}
				else if(i % 2 == 1){
					for(int j = 0; j < layers[i]; j++){
						Convolution(i,j);
						//printf("convolution arch[%d][%d]\n", i, j);
						//Test_showimage(arch[i][j], output_index++);
					}
				}
				else if(i == 6){
					temp = -1;
					Label = -1;
					for(int j = 0; j < layers[i]; j++){
						MLP(i, j);
						//printf("output = %lf\n", arch[6][j].data[0][0]);
						fprintf(fp, "%lf\t", arch[6][j].data[0][0]);

						if ( temp < arch[6][j].data[0][0]){
							temp = arch[6][j].data[0][0];
							Label = j;
						}//Test_showimage(arch[i][j]);
					}
					//printf("result = %d\n", Label);
					fprintf(fp, "\n%d\t", Label);
					fprintf(fp, "%d\n", resultTarget[it]);
					if( Label == resultTarget[it]){
						fprintf(fp, "True\n");
						sprintf(buffer, "Label : %d - %d", Label, resultTarget[it]);
						sprintf(OutName, "loadWeight\\out2\\%d-True.jpg", it+1);
						trueCount++;
					}
					else{
						fprintf(fp, "false\n");
						sprintf(buffer, "Label : %d - %d", Label, resultTarget[it]);
						sprintf(OutName, "loadWeight\\out2\\%d-False.jpg", it+1);
					}
					
					cvPutText(src, buffer, cvPoint(10,10), &font, cvScalarAll(255));
					
					cvSaveImage(OutName, src);
				
				}
				else if(i % 2 == 0){
					for(int j = 0; j < layers[i]; j++){
						Subsampling(i,j);
						//printf("Subsampling arch[%d][%d]\n", i, j);
						//Test_showimage(arch[i][j], output_index++);
					}
				}
				
			}
			cvReleaseImage(&src);			
			/*if( (it+1)%1000 == 0)
			{
				tempCount = trueCount;
				tempCount /=(it+1);
				printf("number = %d\t True Count = %d Percentage = %f\n",
					it+1,trueCount, tempCount*100 );
			}*/
		}
		tempCount = (float)trueCount;
		tempCount /= TARGET_SIZE;
		printf(" %d True Count Number = %d\n",p, trueCount);
		printf(" Percentage = %f\n", tempCount*100);
		fprintf(fp, "\t%Total True Count Number : %d\n", trueCount);
		fprintf(fp, "\tTotal True Percentage : %f\n", tempCount*100);

		fclose(fp);
	}
}

void LoadWeight(int number)
{
	char name[50];
	sprintf(name, "loadWeight\\weight-%d.txt", number);
	FILE *fp = fopen(name,"r");

	for(int i =0; i<5; i++){
		if(i % 2 == 0){
			for(int j = 0; j < layers[i+1]; j++){
				fscanf(fp, "%lf\t", &arch[i+1][j].bias);
				for(int k = 0; k < layers[i]; k++){
					for(int l = 0; l < filter_size[i]; l++){
						for(int m = 0; m < filter_size[i]; m++){
							fscanf(fp, "%lf\t", &weights[i][j][k].filter[l][m]);
						}						
					}										
				}//fprintf(fp,"\n");				
			}
		}
		else{
			for(int j = 0; j < layers[i+1]; j++){
				fscanf(fp, "%lf\t", &weights[i][j][0].beta );
			}//fprintf(fp, "\n");			
		}
	}

	for(int i = 0; i < layers[6]; i++){
		for(int j = 0; j < layers[5]; j++){			
			fscanf(fp, "%lf\t", &weights[5][i][j].beta);
		}		
	}
	fclose(fp);
}
void LoadWeight()
{
	FILE *fp = fopen("loadWeight\\weight.txt","r");
	for(int i =0; i<6; i++){
		for(int j = 0; j < layers[i+1]; j++){
			for(int k = 0; k < layers[i]; k++){
				fscanf(fp, "%lf ", &arch[i+1][j].bias);
				if(i == 6){
					fscanf(fp, "%lf ", &weights[i][j][k].beta);
				}
				else if(i % 2 == 0){
					for(int l = 0; l < weights[i][j][k].height; l++){
						for(int m = 0; m < weights[i][j][k].width; m++){
							fscanf(fp, "%lf ", &weights[i][j][k].filter[l][m]);
						}//fscanf(fp, "\n");						
					}
				}
				else if(i % 2 == 1){
					fscanf(fp, "%lf ", &weights[i][j][k].beta);
				}

			}
		}
	}
}




void init_delta(){
	for(int i = 0; i < 6; i++){
		for(int j = 0; j < layers[i]; j++){
			for(int k = 0; k < arch[i][j].height; k++){
				for(int l = 0; l < arch[i][j].width; l++){
					arch[i][j].delta[k][l] = 0;
				}
			}
		}
	}
}

void MLP(int i, int j){
	arch[i][j].data[0][0] = arch[i][j].bias;	
	for(int k = 0; k < layers[i-1]; k++){
		arch[i][j].data[0][0] += arch[i-1][k].data[0][0] * weights[i-1][j][k].beta;		
	}
	arch[i][j].data[0][0] = tanh(arch[i][j].data[0][0]);
	arch[i][j].fn[0][0] = 1- tanh(arch[i][j].data[0][0])*tanh(arch[i][j].data[0][0]);
}

void Update_weight(Img *arch_before, Img *arch_after, Weight *weight, int mode)
{
	double delta_weight_sum = 0;
	if(mode == 1){
		for(int i = 0; i < weight->height; i++){
			for(int j = 0; j < weight->width; j++){
				delta_weight_sum = 0;
				for(int k = 0; k < arch_after->height; k++){
					for(int l = 0; l < arch_after->width; l++){
						delta_weight_sum += arch_before->data[i+k][j+l] * arch_after->delta[k][l];
					}
				}
				weight->filter[i][j] += delta_weight_sum * learning_gain;
			}
		}
	}
	else if(mode == 2){
		delta_weight_sum = 0;
		for(int i = 0; i < arch_before->height; i+= weight->height)
		{
			for(int j = 0; j < arch_before->width; j+=weight->width)
			{
				for(int k = 0; k < weight->height; k++)
				{
					for(int l = 0; l < weight->width; l++)
					{
						delta_weight_sum += arch_before->data[i+k][j+l] * arch_after->delta[i/weight->height][j/weight->width];
					}
				}
			}
		}
		delta_weight_sum /= (arch_before->height * arch_before->width);
		weight->beta += delta_weight_sum * learning_gain;
	}

}

void Convolution(int i, int j)
{
	int margin = filter_size[i-1];	
	for(int l = 0; l < arch[i-1][0].height - margin + 1; l++){
		for(int m = 0; m < arch[i-1][0].width - margin + 1; m++){
			arch[i][j].data[l][m] = arch[i][j].bias;
			for(int k = 0; k < layers[i-1]; k++){
				for(int n = 0; n < margin; n++){
					for(int o = 0; o < margin; o++){
						arch[i][j].data[l][m] += arch[i-1][k].data[l+n][m+o] * weights[i-1][j][k].filter[n][o];
					}
				}			
			}			
			arch[i][j].data[l][m] = tanh(arch[i][j].data[l][m]);
			arch[i][j].fn[l][m] = 1- tanh(arch[i][j].data[l][m])*tanh(arch[i][j].data[l][m]);
		}
	}
}

void Cal_delta(Img *arch_before, Img *arch_after, Weight *weight, int mode)
{
	//mode = 1, convolutional Layer 
	//mode = 2, subSampling Layer	
	if(mode == 1){
		for(int ii = 0; ii < arch_after->height; ii++){
			for(int jj = 0; jj < arch_after->width; jj++){
				for(int kk = 0; kk < weight->height; kk++){
					for(int ll = 0; ll< weight->width; ll++){
						arch_before->delta[ii+kk][jj+ll] += arch_before->fn[ii+kk][jj+ll] * weight->filter[kk][ll] * arch_after->delta[ii][jj];
					}
				}
			}
		}
	}
	else if(mode ==2){
		for(int i = 0; i < arch_before->height; i+=weight->height){
			for(int j = 0; j < arch_before->width; j+=weight->width){
				for(int k = 0; k < weight->height; k++){
					for(int l = 0; l < weight->width; l++){
 						arch_before->delta[i+k][j+l] += arch_before->fn[i+k][j+l] * weight->beta * arch_after->delta[i/weight->height][j/weight->height];
					}
				}
			}
		}
	}
}

void Subsampling(int i, int j)
{
	int sub_size = filter_size[i-1];
	double sum = 0;
	int num_input = sub_size * sub_size;

	for(int l = 0; l < arch[i-1][j].height; l+=sub_size){
		for(int m = 0; m < arch[i-1][j].width; m+=sub_size){
			sum = 0;
			for(int n = 0; n < sub_size; n++){
				for(int o = 0; o < sub_size; o++){
					sum += arch[i-1][j].data[l+n][m+o];
				}
			}			
			arch[i][j].data[l/sub_size][m/sub_size] = (sum / (double)num_input) + arch[i][j].bias;
			arch[i][j].data[l/sub_size][m/sub_size] *= weights[i-1][j][0].beta;     
			arch[i][j].data[l/sub_size][m/sub_size] = tanh(arch[i][j].data[l/sub_size][m/sub_size]);			
			arch[i][j].fn[l/sub_size][m/sub_size] = 1- tanh(arch[i][j].data[l/sub_size][m/sub_size])*tanh(arch[i][j].data[l/sub_size][m/sub_size]);
		}
	}
}

void Img_alloc(Img* archs){
	archs->data = (double**)calloc(archs->height, sizeof(double*));
	archs->delta = (double**)calloc(archs->height, sizeof(double*));
	archs->fn = (double**)calloc(archs->height, sizeof(double*));
	for(int i = 0; i < archs->height; i++){
		archs->data[i] = (double*)calloc(archs->width, sizeof(double));
		archs->delta[i] = (double*)calloc(archs->width, sizeof(double));
		archs->fn[i] = (double*)calloc(archs->width, sizeof(double));
	}	
	archs->bias = ((rand() % 201) * 0.01) -1;
}

void Save_Weight(int epoch, double energy){
	FILE *fp_save;
	char Name[50];
	sprintf(Name, "weight\\weight-%d-%lf.txt", epoch, energy);
	fp_save = fopen(Name, "w");
	//0, 1
	for (int i = 0; i < 6; i++){
		//3, 2
		for (int j = 0; j < layers[i + 1]; j++){
			//1, 3
			for (int k = 0; k < layers[i]; k++){
				fprintf(fp_save, "%lf ", arch[i + 1][j].bias);
				if (i == 6){
					fprintf(fp_save, "%lf ", weights[i][j][k].beta);
				}
				else if (i % 2 == 0){
					for (int l = 0; l < weights[i][j][k].height; l++){
						for (int m = 0; m < weights[i][j][k].width; m++){
							fprintf(fp_save, "%lf ", weights[i][j][k].filter[l][m]);
						}
						fprintf(fp_save, "\n");
					}
				}
				else if (i % 2 == 1){
					fprintf(fp_save, "%lf ", weights[i][j][k].beta);
				}

			}
		}
	}
	fclose(fp_save);	
}
void Save_Weight(){
	FILE *fp_save;
	fp_save = fopen("save.txt", "w");
	// 0, 1
	for(int i = 0; i < 6; i++){
		//3, 2
		for(int j = 0; j < layers[i+1]; j++){
			//1, 3
			for(int k = 0; k < layers[i]; k++){
				fprintf(fp_save, "%lf ", arch[i+1][j].bias);
				if(i == 6){
					fprintf(fp_save, "%lf ", weights[i][j][k].beta);
				}
				else if(i % 2 == 0){
					for(int l = 0; l < weights[i][j][k].height; l++)
					{
						for(int m = 0; m < weights[i][j][k].width; m++)
						{
							fprintf(fp_save, "%lf ", weights[i][j][k].filter[l][m]);
						}
						fprintf(fp_save, "\n");
					}
				}
				else if(i % 2 == 1){
					fprintf(fp_save, "%lf ", weights[i][j][k].beta);
				}
			}
		}
	}
	fclose(fp_save);
}