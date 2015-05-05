//
//  KRBPNTrainedNetwork.h
//  BPN V2.1.1
//
//  Created by Kalvar on 2014/5/22.
//  Copyright (c) 2014 - 2015年 Kuo-Ming Lin (Kalvar Lin, ilovekalvar@gmail.com). All rights reserved.
//

@interface KRBPNTrainedNetwork : NSObject <NSCoding>

@property (nonatomic, strong) NSMutableArray *inputs;
@property (nonatomic, strong) NSMutableArray *inputWeights;
@property (nonatomic, strong) NSMutableArray *hiddenLayers;
@property (nonatomic, strong) NSMutableArray *allHiddenWeights;
@property (nonatomic, strong) NSMutableArray *allHiddenBiases;
@property (nonatomic, strong) NSMutableArray *outputBiases;
@property (nonatomic, strong) NSArray *outputResults;
@property (nonatomic, strong) NSMutableArray *outputGoals;
@property (nonatomic, assign) CGFloat learningRate;
@property (nonatomic, assign) double convergenceError;
@property (nonatomic, assign) float fOfAlpha;
@property (nonatomic, assign) NSInteger limitGeneration;
@property (nonatomic, assign) NSInteger trainingGeneration;

+(instancetype)sharedNetwork;
-(instancetype)init;

@end
