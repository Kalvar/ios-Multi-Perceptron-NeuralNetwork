//
//  KRANN.h
//  ANN V2.1.4 ( 多層感知倒傳遞類神經網路 ; 本方法使用其中的 EBP 誤差導傳遞類神經網路建構 )
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2015年 Kuo-Ming Lin (Kalvar Lin, ilovekalvar@gmail.com). All rights reserved.
//

/*
 * @ N 層架構
 *   - 輸入層
 *   - 隱藏層 x N
 *   - 輸出層
 */
#import <Foundation/Foundation.h>

/*
 * @ 儲存訓練過後的 ANN Network 數據資料
 */
#import "KRANNTrainedNetwork.h"

/*
 * @ 訓練完成時
 *   - success     : 是否訓練成功
 *   - trainedInfo : 訓練後的 Network 資料
 *   - totalTimes  : 共訓練幾次即達到收斂
 */
typedef void(^KRANNTrainingCompletion)(BOOL success, NSDictionary *trainedInfo, NSInteger totalTimes);

/*
 * @ 每一次的迭代資料
 *   - times       : 訓練到了第幾代
 *   - trainedInfo : 本次訓練的 Network 資料
 */
typedef void(^KRANNPerIteration)(NSInteger times, NSDictionary *trainedInfo);

/*
 * @ 當前訓練的 ANN Network 數據資料
 *   - trainedInfo = @{};
 *      - KRANNTrainedInputWeights   : NSMutableArray, 調整後的輸入層各向量值到第 1 層隱藏層神經元的權重
 *      - KRANNTrainedHiddenLayers   : NSMutableArray, 調整後所有的隱藏層數據
 *      - KRANNTrainedHiddenWeights  : NSMutableArray, 調整後的各隱藏層神經元到輸出層神經元權重值
 *      - KRANNTrainedHiddenBiases   : NSMutableArray, 調整後的各隱藏層神經元偏權值
 *      - KRANNTrainedOutputBiases   : NSMutableArray, 調整後的各輸出層神經元偏權值
 *      - KRANNTrainedOutputResults  : NSArray,        輸出結果
 *      - KRANNTrainedIterations    : NSInteger,      已訓練到第幾代
 *
 */
static NSString *KRANNTrainedInputWeights      = @"KRANNTrainedInputWeights";
static NSString *KRANNTrainedHiddenLayers      = @"KRANNTrainedHiddenLayers";
static NSString *KRANNTrainedHiddenWeights     = @"KRANNTrainedHiddenWeights";
static NSString *KRANNTrainedHiddenBiases      = @"KRANNTrainedHiddenBiases";
static NSString *KRANNTrainedOutputBiases      = @"KRANNTrainedOutputBiases";
static NSString *KRANNTrainedOutputResults     = @"KRANNTrainedOutputResults";
static NSString *KRANNTrainedIterations       = @"KRANNTrainedIterations";

typedef enum KRANNActivationFunctions
{
    //Sigmoid
    KRANNActivationFunctionSigmoid = 0,
    //Tanh
    KRANNActivationFunctionTanh,
    //Fuzzy, still not complete
    KRANNActivationFunctionFuzzy
}KRANNActivationFunctions;

@protocol KRANNDelegate;

@interface KRANN : NSObject
{
    
}

//Setup attribute is strong that 'coz we want to keep the delegate when the training run in the other queue.
@property (nonatomic, strong) id<KRANNDelegate> delegate;

//輸入層各向量值之陣列集合
@property (nonatomic, strong) NSMutableArray *inputs;
//輸入層各向量值到第 1 層隱藏層神經元的權重
@property (nonatomic, strong) NSMutableArray *inputWeights;
//隱藏層神經元到下一層層神經元的權重值 ( 即隱藏層神經元往後指的線權重 )
@property (nonatomic, strong) NSMutableArray *allHiddenWeights;
//隱藏層每一顆 Net 的偏權值
@property (nonatomic, strong) NSMutableArray *allHiddenBiases;
//所有隱藏層的所有神經元設定
@property (nonatomic, strong) NSMutableArray *hiddenLayers;
//要隨機設定幾層隱藏層
@property (nonatomic, assign) NSInteger hiddenLayerCount;

//共有幾層隱藏層
@property (nonatomic, assign) NSInteger countHiddenLayers;
//共有幾顆輸出層的神經元
@property (nonatomic, assign) NSInteger countOutputNets;
//共有幾顆輸入層神經元
@property (nonatomic, assign) NSInteger countInputNets;

//輸出層神經元偏權值
@property (nonatomic, strong) NSMutableArray *outputBiases;
//輸出層的輸出值( 輸出結果 )
@property (nonatomic, strong) NSArray *outputResults;
//所有輸入向量( 每一組訓練資料 )的各別輸出期望值
@property (nonatomic, strong) NSMutableArray *outputGoals;
//學習速率
@property (nonatomic, assign) CGFloat learningRate;
//收斂誤差值 ( 10^-3, 10^-6 )
@property (nonatomic, assign) double convergenceError;
//活化函式的 Alpha Value ( LMS 的坡度 )
@property (nonatomic, assign) float fOfAlpha;

//訓練迭代次數上限
@property (nonatomic, assign) NSInteger limitIteration;
//目前訓練到第幾代
@property (nonatomic, assign) NSInteger trainingIteration;
//是否正在訓練中
@property (nonatomic, assign) BOOL isTraining;
//當前訓練後的資料
@property (nonatomic, strong) NSDictionary *trainedInfo;
//取出儲存在 NSUserDefaults 裡訓練後的完整 ANN Network 數據資料
@property (nonatomic, readwrite) KRANNTrainedNetwork *trainedNetwork;
//Use which f(x) the activiation function
@property (nonatomic, assign) KRANNActivationFunctions activationFunction;

//@property (nonatomic, assign) BOOL openDebug;

@property (nonatomic, copy) KRANNTrainingCompletion trainingCompletion;
@property (nonatomic, copy) KRANNPerIteration perIteration;

+(instancetype)sharedNetwork;
-(instancetype)init;

#pragma --mark Settings Public Methods
-(void)addPatterns:(NSArray *)_patterns outputGoals:(NSArray *)_goals;
-(void)addPatternWeights:(NSArray *)_weights;
-(void)addHiddenLayerAtIndex:(int)_layerIndex netBias:(float)_netBias netWeights:(NSArray *)_netWeights;
-(void)addOutputBiases:(NSArray *)_biases;

#pragma --mark Setting Paramaters Public Methods
-(double)randomMax:(double)_maxValue min:(double)_minValue;
-(NSInteger)evaluateHiddenLayerNumbers;
-(void)randomHiddenWeightsWithTotalLayers:(int)_totalLayers;
-(void)randomInputWeights;
-(void)randomWeights;

#pragma --mark Training Public Methods
-(void)training;
-(void)trainingBySave;
-(void)trainingByRandomSettings;
-(void)trainingByRandomWithSave;
-(void)trainingWithAddPatterns:(NSArray *)_patterns outputGoals:(NSArray *)_goals;
-(void)pause;
-(void)continueTraining;
-(void)reset;
-(void)restart;
-(void)directOutputAtInputs:(NSArray *)_rawInputs completion:(void(^)())_completion;
-(void)directOutputAtInputs:(NSArray *)_rawInputs;

#pragma --mark Trained Network Public Methods
-(void)saveNetwork;
-(void)removeNetwork;
-(void)recoverNetwork:(KRANNTrainedNetwork *)_recoverNetworks;
-(void)recoverNetwork;

#pragma --mark Blocks
-(void)setTrainingCompletion:(KRANNTrainingCompletion)_theBlock;
-(void)setPerIteration:(KRANNPerIteration)_theBlock;

@end

@protocol KRANNDelegate <NSObject>

@optional
-(void)krANNDidTrainFinished:(KRANN *)krAnn trainedInfo:(NSDictionary *)trainedInfo totalTimes:(NSInteger)totalTimes;
-(void)krANNPerIteration:(KRANN*)krAnn trainedInfo:(NSDictionary *)trainedInfo times:(NSInteger)times;

@end
