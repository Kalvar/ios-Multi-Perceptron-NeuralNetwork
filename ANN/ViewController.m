//
//  ViewController.m
//  ANN V2.1.4
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2015年 Kuo-Ming Lin (Kalvar Lin, ilovekalvar@gmail.com). All rights reserved.
//

#import "ViewController.h"
#import "KRANN.h"

@interface ViewController ()<KRANNDelegate>

@property (nonatomic, strong) KRANN *_krANN;

@end

@implementation ViewController

@synthesize _krANN;

//Setups any detail, and 2 outputs, you could set more outputs.
-(void)useSample1
{
    _krANN.activationFunction = KRANNActivationFunctionSigmoid;
    //各輸入向量陣列值 & 每一筆輸入向量的期望值( 輸出期望 )，因使用 S 形轉換函數，故 Input 值域須為 [0, 1]，輸出目標為 [0, 1]
    //Add the patterns, the weights connect with hidden layer, the output targets
    [_krANN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoals:@[@0.7f, @0.8f]];  //Pattern 1, net 1, 2, 3, 4, the output layer is 2 nets
    [_krANN addPatterns:@[@0, @1, @0.3, @0.9] outputGoals:@[@0.1f, @0.1f]];    //Pattern 2, same as pattern 1
    [_krANN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoals:@[@0.95f, @0.9f]]; //Pattern 3, same as pattern 1
    
    //輸入層各向量值到隱藏層神經元的權重 ( 連結同一個 Net 的就一組一組分開，有幾個 Hidden Net 就會有幾組 )
    //Add pattern-weights in Input layer to first Hidden Layer
    [_krANN addPatternWeights:@[@0.2, @-0.3]]; //W15, W16
    [_krANN addPatternWeights:@[@0.4, @0.1]];  //W25, W26
    [_krANN addPatternWeights:@[@-0.5, @0.2]]; //W35, W36
    [_krANN addPatternWeights:@[@-0.1, @0.3]]; //W45, W46
    
    //隱藏層神經元的偏權值 & 隱藏層神經元到輸出層神經元的權重值
    //Add Hidden Layers the biases of nets and the output weights in connect with output layer.
    [_krANN addHiddenLayerAtIndex:0 netBias:-0.4f netWeights:@[@-0.3f, @0.2f, @0.1f]]; //Net 5
    [_krANN addHiddenLayerAtIndex:0 netBias:0.2f netWeights:@[@-0.2f, @0.5f, @-0.1f]];  //Net 6
    
    [_krANN addHiddenLayerAtIndex:1 netBias:0.1f netWeights:@[@0.1f, @0.3f]];   //Net 7
    [_krANN addHiddenLayerAtIndex:1 netBias:0.25f netWeights:@[@0.2f, @0.1f]];  //Net 8
    [_krANN addHiddenLayerAtIndex:1 netBias:0.3f netWeights:@[@0.3f, @-0.4f]];  //Net 9
    
    [_krANN addHiddenLayerAtIndex:2 netBias:-0.25f netWeights:@[@0.4f, @0.3f]];  //Net 10
    [_krANN addHiddenLayerAtIndex:2 netBias:0.15f netWeights:@[@0.1f, @-0.2f]];  //Net 11
    
    //輸出層神經元偏權值, Net 12, Net 13
    //Add the output layer biases
    [_krANN addOutputBiases:@[@0.0f, @0.1f]];
    
    __block typeof(_krANN) _weakKRANN = _krANN;
    //訓練完成時( Training complete )
    [_krANN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKRANN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKRANN recoverNetwork];
            [_weakKRANN directOutputAtInputs:@[@1, @0.1, @0.5, @0.2]];
        }
    }];
    
    [_krANN training];
}

//Only setups patterns and output goals, and 1 output.
-(void)useSample2
{
    //Use f(x)=tanh(x)
    _krANN.activationFunction = KRANNActivationFunctionTanh;
    //How many hidden layers
    _krANN.hiddenLayerCount   = 0; //0 means let system decide the hidden layers number.
    
    [_krANN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoals:@[@0.7f]];    //Pattern 1, net 1, 2, 3, 4, and 1 output
    [_krANN addPatterns:@[@0, @-0.8, @0.3, @-0.9] outputGoals:@[@-0.1f]]; //Pattern 2, same as pattern 1
    [_krANN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoals:@[@0.9f]];    //Pattern 3, same as pattern 1
    
    __block typeof(_krANN) _weakKRANN = _krANN;
    //訓練完成時( Training complete )
    [_krANN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKRANN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKRANN recoverNetwork];
            [_weakKRANN directOutputAtInputs:@[@0, @-0.8, @0.3, @-0.9]];
        }
    }];
    
    [_krANN trainingByRandomWithSave];
}

//To learn and verify numbers 0 to 9. And only setups patterns and output goals, and 10 outputs.
-(void)useSample3
{
    _krANN.activationFunction = KRANNActivationFunctionSigmoid;
    //How many hidden layers, if you set 0 that means let system decide the hidden layers number.
    _krANN.hiddenLayerCount = 1;
    
    //1
    [_krANN addPatterns:@[@0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@1, @0, @0, @0, @0, @0, @0, @0, @0, @0]];
    //2
    [_krANN addPatterns:@[@1, @0, @0, @0,
                          @1, @1, @1, @1,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @1, @1, @1, @1,
                          @0, @0, @0, @1]
            outputGoals:@[@0, @1, @0, @0, @0, @0, @0, @0, @0, @0]];
    //3
    [_krANN addPatterns:@[@1, @0, @0, @0,
                          @1, @0, @0, @0,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @1, @0, @0, @0, @0, @0, @0, @0]];
    //4
    [_krANN addPatterns:@[@1, @1, @1, @1,
                          @1, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @1, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @1, @0,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @1, @0, @0, @0, @0, @0, @0]];
    //5
    [_krANN addPatterns:@[@1, @1, @1, @1,
                          @1, @0, @0, @0,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @1, @0, @0, @0, @0, @0]];
    //6
    [_krANN addPatterns:@[@1, @1, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @1, @0, @0, @0, @0]];
    //7
    [_krANN addPatterns:@[@1, @0, @0, @0,
                          @0, @0, @0, @0,
                          @0, @1, @0, @0,
                          @0, @0, @0, @0,
                          @0, @0, @1, @0,
                          @0, @0, @0, @0,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @0, @1, @0, @0, @0]];
    //8
    [_krANN addPatterns:@[@1, @1, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @1, @0, @0]];
    //9
    [_krANN addPatterns:@[@1, @1, @1, @1,
                          @1, @0, @0, @0,
                          @0, @1, @0, @0,
                          @0, @1, @0, @0,
                          @0, @0, @1, @0,
                          @0, @0, @1, @0,
                          @0, @0, @0, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @0, @1, @0]];
    //0
    [_krANN addPatterns:@[@1, @1, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @0, @0,
                          @0, @0, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @0, @0,
                          @0, @0, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @0, @0, @1]];
    
    __block typeof(_krANN) _weakKRANN = _krANN;
    //訓練完成時( Training complete )
    [_krANN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKRANN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKRANN recoverNetwork];
            //Verified number " 7 ", and it has some defects.
            [_weakKRANN directOutputAtInputs:@[@1, @1, @1, @0,
                                               @0, @0, @0, @0,
                                               @0, @1, @0, @0,
                                               @0, @0, @0, @0,
                                               @0, @0, @1, @0,
                                               @0, @0, @0, @0,
                                               @0, @0, @0, @1,
                                               @1, @1, @1, @1,
                                               @1, @1, @1, @1]];
            
        }
    }];
    
    [_krANN trainingByRandomSettings];
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    
	_krANN          = [KRANN sharedNetwork];
    //_krANN.delegate = self;
    
    //學習速率
    _krANN.learningRate     = 0.8f;
    //收斂誤差值 ( Normally is 10^-3 or 10^-6 )
    _krANN.convergenceError = 0.000001f;
    //限制迭代次數
    _krANN.limitIteration  = 100;
    
    //每一次的迭代( Per iteration-training )
    [_krANN setPerIteration:^(NSInteger times, NSDictionary *trainedInfo)
    {
        NSLog(@"Iteration times : %i", times);
        //NSLog(@"Iteration result : %f\n\n\n", [trainedInfo objectForKey:KRANNTrainedOutputResults]);
    }];
    
    //[self useSample1];
    //[self useSample2];
    [self useSample3];
    
    //Remove your testing trained-network records.
    //[_krANN removeNetwork];
    
    //Start the training, and random the weights, biases, if you use this method that you won't need to setup any weights and biases before.
    //Random means let network to auto setup inputWeights, hiddenBiases, hiddenWeights values.
    //[_krANN trainingByRandomSettings];
    //As above said, then it will be saved the trained network after done.
    //[_krANN trainingByRandomWithSave];
    
    //Start the training network, and it won't be saving the trained-network when finished.
    //[_krANN training];
    
    //Start the training network, and it will auto-saving the trained-network when finished.
    //[_krANN trainingBySave];
    
    //If you wanna pause the training.
    //[_krANN pause];
    
    //If you wanna continue the paused training.
    //[_krANN continueTraining];
    
    //If you wanna reset the network back to initial situation.
    //[_krANN reset];
    
    //When the training finished, to save the trained-network into NSUserDefaults.
    //[_krANN saveNetwork];
    
    //If you wanna recover the trained-network data.
    //[_krANN recoverNetwork];
    
}

- (void)didReceiveMemoryWarning

{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

#pragma --mark KRANNDelegate
/*
-(void)krANNDidTrainFinished:(KRANN *)krAnn trainedInfo:(NSDictionary *)trainedInfo totalTimes:(NSInteger)totalTimes
{
    NSLog(@"Use trained-network to direct output : %@", krAnn.outputResults);
}

-(void)krANNPerIteration:(KRANN *)krAnn trainedInfo:(NSDictionary *)trainedInfo times:(NSInteger)times
{
    NSLog(@"Iteration times : %i", times);
}
 */


@end
