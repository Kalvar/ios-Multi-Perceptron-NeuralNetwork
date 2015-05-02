//
//  ViewController.m
//  BPN V2.1
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2015年 Kuo-Ming Lin (Kalvar Lin, ilovekalvar@gmail.com). All rights reserved.
//

#import "ViewController.h"
#import "KRBPN.h"

@interface ViewController ()<KRBPNDelegate>

@property (nonatomic, strong) KRBPN *_krBPN;

@end

@implementation ViewController

@synthesize _krBPN;

//Setups any detail, and 2 outputs, you could set more outputs.
-(void)useSample1
{
    //各輸入向量陣列值 & 每一筆輸入向量的期望值( 輸出期望 )，因使用 S 形轉換函數，故 Input 值域須為 [0, 1]，輸出目標為 [0, 1]
    //Add the patterns, the weights connect with hidden layer, the output targets
    [_krBPN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoals:@[@0.7f, @0.8f]];  //Pattern 1, net 1, 2, 3, 4, the output layer is 2 nets
    [_krBPN addPatterns:@[@0, @1, @0.3, @0.9] outputGoals:@[@0.1f, @0.1f]];    //Pattern 2, same as pattern 1
    [_krBPN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoals:@[@0.95f, @0.9f]]; //Pattern 3, same as pattern 1
    
    //輸入層各向量值到隱藏層神經元的權重 ( 連結同一個 Net 的就一組一組分開，有幾個 Hidden Net 就會有幾組 )
    //Add pattern-weights in Input layer to first Hidden Layer
    [_krBPN addPatternWeights:@[@0.2, @-0.3]]; //W15, W16
    [_krBPN addPatternWeights:@[@0.4, @0.1]];  //W25, W26
    [_krBPN addPatternWeights:@[@-0.5, @0.2]]; //W35, W36
    [_krBPN addPatternWeights:@[@-0.1, @0.3]]; //W45, W46
    
    //隱藏層神經元的偏權值 & 隱藏層神經元到輸出層神經元的權重值
    //Add Hidden Layers the biases of nets and the output weights in connect with output layer.
    [_krBPN addHiddenLayerAtIndex:0 netBias:-0.4f netWeights:@[@-0.3f, @0.2f, @0.1f]]; //Net 5
    [_krBPN addHiddenLayerAtIndex:0 netBias:0.2f netWeights:@[@-0.2f, @0.5f, @-0.1f]];  //Net 6
    
    [_krBPN addHiddenLayerAtIndex:1 netBias:0.1f netWeights:@[@0.1f, @0.3f]];   //Net 7
    [_krBPN addHiddenLayerAtIndex:1 netBias:0.25f netWeights:@[@0.2f, @0.1f]];  //Net 8
    [_krBPN addHiddenLayerAtIndex:1 netBias:0.3f netWeights:@[@0.3f, @-0.4f]];  //Net 9
    
    [_krBPN addHiddenLayerAtIndex:2 netBias:-0.25f netWeights:@[@0.4f, @0.3f]];  //Net 10
    [_krBPN addHiddenLayerAtIndex:2 netBias:0.15f netWeights:@[@0.1f, @-0.2f]];  //Net 11
    
    //輸出層神經元偏權值, Net 12, Net 13
    //Add the output layer biases
    [_krBPN addOutputBiases:@[@0.0f, @0.1f]];
    
    __block typeof(_krBPN) _weakKrBPN = _krBPN;
    //訓練完成時( Training complete )
    [_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKrBPN recoverNetwork];
            [_weakKrBPN directOutputAtInputs:@[@1, @0.1, @0.5, @0.2]];
        }
    }];
    
    [_krBPN training];
    //[_krBPN trainingSave];
}

//Only setups patterns and output goals, and 1 output.
-(void)useSample2
{
    //How many hidden layers
    _krBPN.hiddenLayerCount = 0; //0 means let system decide the hidden layers number.
    
    [_krBPN addPatterns:@[@1, @0.1, @0.5, @0.2] outputGoals:@[@0.7f]]; //Pattern 1, net 1, 2, 3, 4, and 1 output
    [_krBPN addPatterns:@[@0, @0.8, @0.3, @0.9] outputGoals:@[@0.1f]]; //Pattern 2, same as pattern 1
    [_krBPN addPatterns:@[@1, @0.3, @0.1, @0.4] outputGoals:@[@0.9f]]; //Pattern 3, same as pattern 1
    
    __block typeof(_krBPN) _weakKrBPN = _krBPN;
    //訓練完成時( Training complete )
    [_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKrBPN recoverNetwork];
            [_weakKrBPN directOutputAtInputs:@[@1, @0.1, @0.5, @0.2]];
        }
    }];
    
    [_krBPN trainingRandom];
    //[_krBPN trainingRandomAndSave];
}

//To learn and verify numbers 0 to 9. And only setups patterns and output goals, and 10 outputs.
-(void)useSample3
{
    //How many hidden layers, if you set 0 that means let system decide the hidden layers number.
    _krBPN.hiddenLayerCount = 0;
    
    //1
    [_krBPN addPatterns:@[@0, @0, @0, @0,
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
    [_krBPN addPatterns:@[@1, @0, @0, @0,
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
    [_krBPN addPatterns:@[@1, @0, @0, @0,
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
    [_krBPN addPatterns:@[@1, @1, @1, @1,
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
    [_krBPN addPatterns:@[@1, @1, @1, @1,
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
    [_krBPN addPatterns:@[@1, @1, @1, @1,
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
    [_krBPN addPatterns:@[@1, @0, @0, @0,
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
    [_krBPN addPatterns:@[@1, @1, @1, @1,
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
    [_krBPN addPatterns:@[@1, @1, @1, @1,
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
    [_krBPN addPatterns:@[@1, @1, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @0, @0,
                          @0, @0, @0, @0,
                          @0, @1, @1, @0,
                          @0, @0, @0, @0,
                          @0, @0, @1, @1,
                          @1, @1, @1, @1,
                          @1, @1, @1, @1]
            outputGoals:@[@0, @0, @0, @0, @0, @0, @0, @0, @0, @1]];
    
    __block typeof(_krBPN) _weakKrBPN = _krBPN;
    //訓練完成時( Training complete )
    [_krBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
        if( success )
        {
            NSLog(@"Training done with total times : %i", totalTimes);
            NSLog(@"TrainedInfo 1 : %@", trainedInfo);
            
            //Start in checking the network is correctly trained.
            NSLog(@"======== Start in Verification ========");
            [_weakKrBPN setTrainingCompletion:^(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes){
                NSLog(@"Training done with total times : %i", totalTimes);
                NSLog(@"TrainedInfo 2 : %@", trainedInfo);
            }];
            
            [_weakKrBPN recoverNetwork];
            //Verified number " 7 ", and it has some defects.
            [_weakKrBPN directOutputAtInputs:@[@1, @1, @1, @0,
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
    
    [_krBPN trainingRandom];
    //[_krBPN trainingRandomAndSave];
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    
	_krBPN          = [KRBPN sharedNetwork];
    //_krBPN.delegate = self;
    
    //學習速率
    _krBPN.learningRate     = 0.8f;
    //收斂誤差值 ( Normally is 10^-3 or 10^-6 )
    _krBPN.convergenceError = 0.000001f;
    //限制迭代次數
    _krBPN.limitGeneration  = 5000;
    
    //每一次的迭代( Every generation-training )
    [_krBPN setEachGeneration:^(NSInteger times, NSDictionary *trainedInfo)
    {
        NSLog(@"Generation times : %i", times);
        //NSLog(@"Generation result : %f\n\n\n", [trainedInfo objectForKey:KRBPNTrainedOutputResults]);
    }];
    
    //[self useSample1];
    //[self useSample2];
    [self useSample3];
    
    //Remove your testing trained-network records.
    //[_krBPN removeNetwork];
    
    //Start the training, and random the weights, biases, if you use this method that you won't need to setup any weights and biases before.
    //Random means let network to auto setup inputWeights, hiddenBiases, hiddenWeights values.
    //[_krBPN trainingRandom];
    //As above said, then it will be saved the trained network after done.
    //[_krBPN trainingRandomAndSave];
    
    //Start the training network, and it won't be saving the trained-network when finished.
    //[_krBPN training];
    
    //Start the training network, and it will auto-saving the trained-network when finished.
    //[_krBPN trainingSave];
    
    //If you wanna pause the training.
    //[_krBPN pause];
    
    //If you wanna continue the paused training.
    //[_krBPN continueTraining];
    
    //If you wanna reset the network back to initial situation.
    //[_krBPN reset];
    
    //When the training finished, to save the trained-network into NSUserDefaults.
    //[_krBPN saveNetwork];
    
    //If you wanna recover the trained-network data.
    //[_krBPN recoverNetwork];
    //Or you wanna use the KRBPNTrainedNetwork object to recover the training data.
    /*
    KRBPNTrainedNetwork *_trainedNetwork = [[KRBPNTrainedNetwork alloc] init];
    _trainedNetwork.inputs = [NSMutableArray arrayWithObjects:
                              @[@1],
                              @[@0],
                              @[@1],
                              nil];
    [_krBPN recoverNetwork:_trainedNetwork];
    */
    
    //To remove the saved trained-network.
    //[_krBPN removeNetwork];
}

- (void)didReceiveMemoryWarning

{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

#pragma --mark KRBPNDelegate
/*
-(void)krBPNDidTrainFinished:(KRBPN *)krBPN trainedInfo:(NSDictionary *)trainedInfo totalTimes:(NSInteger)totalTimes
{
    NSLog(@"Use trained-network to direct output : %@", krBPN.outputResults);
}

-(void)krBPNEachGeneration:(KRBPN *)krBPN trainedInfo:(NSDictionary *)trainedInfo times:(NSInteger)times
{
    NSLog(@"Generation times : %i", times);
}
 */


@end

