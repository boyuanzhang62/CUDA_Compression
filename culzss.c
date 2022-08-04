/***************************************************************************
 *          Lempel, Ziv, Storer, and Szymanski Encoding and Decoding on CUDA
 *
 *
 ****************************************************************************
 *          CUDA LZSS 
 *   Authors  : Adnan Ozsoy, Martin Swany,Indiana University - Bloomington
 *   Date    : April 11, 2011
 
 ****************************************************************************
 
         Copyright 2011 Adnan Ozsoy, Martin Swany, Indiana University - Bloomington
 
         Licensed under the Apache License, Version 2.0 (the "License");
         you may not use this file except in compliance with the License.
         You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
         Unless required by applicable law or agreed to in writing, software
         distributed under the License is distributed on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
         See the License for the specific language governing permissions and
         limitations under the License.
 ****************************************************************************/
 
 /***************************************************************************
 * Code is adopted from below source
 *
 * LZSS: An ANSI C LZss Encoding/Decoding Routine
 * Copyright (C) 2003 by Michael Dipperstein (mdipper@cs.ucsb.edu)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 ***************************************************************************/

 #include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "culzss.h"
#include <sys/time.h>
#include <string.h>

pthread_t congpu, constr, concpu, consend;

int loopnum=0;
int maxiterations=0;
int numblocks=0;
int blocksize=0;
char * outputfilename;
unsigned int * bookkeeping;

int exit_signal = 0;
extern int intervalSize;

void printBuffer(unsigned char* arr){
	printf("this is front 16 values of input buffer:\n");
	for(int byind = 0; byind < 16; byind ++){
		printf("%d\t", arr[byind]);
	}
	printf("\n");
}
void printBufferOut(unsigned char* arr){
	printf("this is front 32 values of output buffer:\n");
	for(int byind = 0; byind < 16; byind ++){
		printf("%d\t", arr[byind * 2]);
	}
	printf("\n");
	// for(int byind = 0; byind < 16; byind ++){
	// 	printf("%d\t", arr[byind * 2 + 1]);
	// }
	// printf("\n");
}
int getloopcount(){
	return loopnum;
}

void printStatistics(unsigned int* statisticOfMatch, int size){
	printf("----------------------match statistic----------------------\n");
	for(int i = 2; i < size; i++){
		printf("length: %d num: %d\n", i+1, statisticOfMatch[i]);
	}
	printf("----------------------match statistic----------------------\n");
}

void *gpu_consumer (void *q)
{
	queue *fifo;
	int i, d;
	int success=0;
	int interval = intervalSize;
	fifo = (queue *)q;
	int comp_length=0;
	fifo->in_d = initGPUmem((int)blocksize*2 * maxiterations);
	fifo->out_d = initGPUmem((int)blocksize*2 * maxiterations);
	
	for (i = 0; i < maxiterations; i++) {
		success=compression_kernel_wrapper(fifo->encodedHostMemory + i * 2 * blocksize, blocksize, fifo->bufout[fifo->headGC], 
										0, 0, 256, 0,fifo->headGC, fifo->in_d + i * blocksize * 2, fifo->out_d + i * blocksize * 2, interval);
		if(!success){
			printf("Compression failed. Success %d\n",success);
		}
		cudaDeviceSynchronize();
	}
	return (NULL);
}


void *cpu_consumer (void *q)
{
	int i;
	int success=0;
	queue *fifo;	
	fifo = (queue *)q;
	int comp_length=0;
	unsigned char * bckpbuf;
	bckpbuf = (unsigned char *)malloc(sizeof(unsigned char)*blocksize);
    double encodeKernelTime = 0;

	// fifo->encodedHostMemory = (unsigned char *)malloc(maxiterations * blocksize * 2);
	fifo->encodedHostSize = (int *)malloc(sizeof(int) * maxiterations);
	for(int tmpi = 0; tmpi < maxiterations; tmpi ++){
		fifo->encodedHostSize[tmpi] = 0;
	}
	fifo->deviceHeader = initGPUmem(sizeof(int) * blocksize * maxiterations / PCKTSIZE);
	fifo->hostHeader = (int *)malloc(sizeof(int) * blocksize * maxiterations / PCKTSIZE);
	
	aftercompression_wrapper(fifo->in_d, fifo->out_d, fifo->encodedHostMemory, fifo->encodedHostSize, \
							 fifo->deviceHeader, fifo->hostHeader, blocksize, maxiterations, &encodeKernelTime, blocksize);

    printf("encode kernel took: %lf milliseconds\n", encodeKernelTime);
	deleteGPUmem(fifo->in_d);
	deleteGPUmem(fifo->out_d);
	deleteGPUmem(fifo->deviceHeader);
	return (NULL);
}

void *cpu_sender (void *q)
{
	FILE *outFile;

	int i;
	int success=0;
	queue *fifo;	
	fifo = (queue *)q;
	if(outputfilename!=NULL)
		outFile = fopen(outputfilename, "wb");
	else
		outFile = fopen("compressed.dat", "wb");
	int size=0;

	fwrite(bookkeeping, sizeof(unsigned int), maxiterations+2, outFile);
		
	for (i = 0; i < maxiterations; i++) 
	{
		size = fifo->encodedHostSize[i];
		bookkeeping[i + 2] = size + ((i==0)?0:bookkeeping[i+1]);

		fwrite(fifo->encodedHostMemory[blocksize * i * 2], size, 1, outFile);
		loopnum++;
	}
	fseek ( outFile , 0 , SEEK_SET );
	fwrite(bookkeeping, sizeof(unsigned int), maxiterations+2, outFile);
	fclose(outFile);
	return (NULL);
}



queue *queueInit (int maxit,int numb,int bsize)
{
	queue *q;
	maxiterations=maxit;
	numblocks=numb;
	blocksize=bsize;
	
	q = (queue *)malloc (sizeof (queue));
	if (q == NULL) return (NULL);

	int i;
	//alloc bufs
	unsigned char ** buffer;
	unsigned char ** bufferout;
	
	
	/*  allocate storage for an array of pointers */
	buffer = (unsigned char **)malloc((numblocks) * sizeof(unsigned char *));
	if (buffer == NULL) {
		printf("Error: malloc could not allocate buffer\n");
		return;
	}
	bufferout = (unsigned char **)malloc((numblocks) * sizeof(unsigned char *));
	if (bufferout == NULL) {
		printf("Error: malloc could not allocate bufferout\n");
		return;
	}
	  
	/* for each pointer, allocate storage for an array of chars */
	for (i = 0; i < (numblocks); i++) {
		//buffer[i] = (unsigned char *)malloc(blocksize * sizeof(unsigned char));
		buffer[i] = (unsigned char *)initCPUmem(blocksize * sizeof(unsigned char));
		printf("blocksize: %d\n", blocksize);
		if (buffer[i] == NULL) {printf ("Memory error, buffer"); exit (2);}
	}
	for (i = 0; i < (numblocks); i++) {
		//bufferout[i] = (unsigned char *)malloc(blocksize * 2 * sizeof(unsigned char));
		bufferout[i] = (unsigned char *)initCPUmem(blocksize * 2 * sizeof(unsigned char));
		if (bufferout[i] == NULL) {printf ("Memory error, bufferout"); exit (2);}
	}
	
	q->buf = buffer;
	q->bufout = bufferout;
	
	q->headPG = 0;
	q->headGC = 0;
	q->headCS = 0;
	q->headSP = 0;
	
	q->outsize = (int *)malloc(sizeof(int)*numblocks);
	
	q->ledger = (int *)malloc((numblocks) * sizeof(int));
	if (q->ledger == NULL) {
		printf("Error: malloc could not allocate q->ledger\n");
		return;
	}

	for (i = 0; i < (numblocks); i++) {
		q->ledger[i] = 0;
	}
		
	q->mut = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t));
	pthread_mutex_init (q->mut, NULL);
	
	q->produced = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->produced, NULL);
	q->compressed = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->compressed, NULL);	
	q->sendready = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->sendready, NULL);	
	q->sent = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->sent, NULL);
	
	return (q);
}


void signalExitThreads()
{
	exit_signal++;
	while(! (exit_signal > 3));
}


void queueDelete (queue *q)
{
	int i =0;
	
	signalExitThreads();
	
	pthread_mutex_destroy (q->mut);
	free (q->mut);	
	pthread_cond_destroy (q->produced);
	free (q->produced);
	pthread_cond_destroy (q->compressed);
	free (q->compressed);	
	pthread_cond_destroy (q->sendready);
	free (q->sendready);	
	pthread_cond_destroy (q->sent);
	free (q->sent);

	
	for (i = 0; i < (numblocks); i++) {
		deleteCPUmem(q->bufout[i]);	
		deleteCPUmem(q->buf[i]);
	}
	
	deleteGPUStreams();
	
	free(q->buf);
	free(q->bufout);
	free(q->ledger);	
	free(q->outsize);
	free(q->encodedHostMemory);	
	free(q->encodedHostSize);
	free(q->hostHeader);
	free (q);
	
	
	resetGPU();

}


void  init_compression(queue * fifo,int maxit,int numb,int bsize, char * filename, unsigned int * book)
{
	maxiterations=maxit;
	numblocks=numb;
	blocksize=bsize;
	outputfilename = filename;
	bookkeeping = book;
	printf("Initializing the GPU\n");
	initGPU();
	//create consumer threades
	gpu_consumer(fifo);
	cpu_consumer(fifo);
	cpu_sender(fifo);
	
	
	
	return;
}

void join_comp_threads()
{	
	pthread_join (congpu, NULL);
	pthread_join (concpu, NULL);
	pthread_join (consend, NULL);
	exit_signal = 3;
}


