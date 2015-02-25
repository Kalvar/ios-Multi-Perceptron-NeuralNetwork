//
//  KRBPN.m
//  BPN V2.0
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2015年 Kuo-Ming Lin (Kalvar Lin, ilovekalvar@gmail.com). All rights reserved.
//

#import "KRBPN.h"
#import "KRBPN+NSUserDefaults.h"

//限制每層隱藏層最大的神經元總數
#define MAX_HIDDEN_NETS_COUNT  20
//最多只給幾層隱藏層的建議
#define MAX_HIDDEN_LAYER_COUNT 10

/*
 * @ 2 種常用的 BPN 模式
 *
 *   - 1. 全部 100 筆都跑完後，總和全部誤差值，再一次回推調整權重與偏權值，這樣才算 1 個迭代。
 *   - 2. 一邊輸入就一邊調整權重與偏權值，一樣要跑完全部 100 筆後，才算 1 個迭代 ( EBP, 誤差導傳遞 )。
 *
 *   2 種都是在每 1 個迭代結束後，判斷 Output Error 是否有達到收斂，如未收斂，就繼續重頭運算，如收斂，就停止訓練 ( 流程參考 P.131 )。
 *
 * @ EBP ( 誤差導傳遞 ) 的流程
 *
 *   - 將 BPN 做一改進，在輸入每一筆的訓練資料時，就「一邊調整權重與偏權值」，直到所有的訓練樣本( 例如 100 筆 )都訓練完後，
 *     才去判斷是否有達到收斂誤差的標準，如有，才停止網路訓練，如果沒有，就重新再代入 100 筆，跑遞迴重新運算一次。
 *
 * @ Knowledge
 *   - Line Weights means 關聯程度 ; 分配 & 關聯
 *     - Line Weight 是用來分配(dispatch)決定要分到哪一個 Net 是最好的
 *   - Nets Threshold (Biases) means 認知程度 ; 認知 & 訓練
 *   - Line + Net = 數據 -> 關聯 -> 分配 -> 訓練 -> 認知 -> 結果 ; 更簡單說 = 關聯 -> 誰懂 -> 結論
 *   - 舉個例，以單純 I/O 分類器，無修正模式而言 : 
 *     - 正妹 -> 可愛 / 性感 -> 宅宅們各自拍不同部位照片(?) -> 宅宅交換與交叉辨論 -> 達成共識 -> 決定是否為新一代女神。
 *
 */
static NSString *_kOriginalInputs           = @"_kOriginalInputs";
static NSString *_kOriginalInputWeights     = @"_kOriginalInputWeights";
static NSString *_kOriginalHiddenLayers     = @"_kOriginalHiddenLayers";
static NSString *_kOriginalAllHiddenWeights = @"_kOriginalAllHiddenWeights";
static NSString *_kOriginalAllHiddenBiases  = @"_kOriginalAllHiddenBiases";
static NSString *_kOriginalOutputBias       = @"_kOriginalOutputBias";
static NSString *_kOriginalOutputResults    = @"_kOriginalOutputResults";
static NSString *_kOriginalOutputGoals      = @"_kOriginalOutputGoals";
static NSString *_kOriginalLearningRate     = @"_kOriginalLearningRate";
static NSString *_kOriginalConvergenceError = @"_kOriginalConvergenceError";
static NSString *_kOriginalFOfAlpha         = @"_kOriginalFOfAlpha";
static NSString *_kOriginalLimitGenerations = @"_kOriginalLimitGenerations";
//static NSString *_kOriginalMaxMultiple      = @"_kOriginalMaxMultiple";

static NSString *_kTrainedNetworkInfo       = @"kTrainedNetworkInfo";

@interface KRBPN ()

//隱藏層的輸出值
@property (nonatomic, strong) NSArray *_hiddenOutputs;
//當前資料的輸出期望值
@property (nonatomic, assign) double _goalValue;
//輸出層的誤差值
@property (nonatomic, strong) NSArray *_outputErrors;
//是否強制中止訓練
@property (nonatomic, assign) BOOL _forceStopTraining;
//原來的設定值
@property (nonatomic, strong) NSMutableDictionary *_originalParameters;
//訓練完就儲存至 NSUserDefaults 裡
@property (nonatomic, assign) BOOL _isDoneSave;
//記錄當前訓練到哪一組 Input 數據
@property (nonatomic, assign) NSInteger _patternIndex;
//在訓練 goalValue 且其值不在 -1.0f ~ 1.0f 之間時，就使用本值進行相除與回乘原同類型值的動作
@property (nonatomic, assign) NSInteger _maxMultiple;

@end

@implementation KRBPN (fixInitials)

-(void)_resetTrainedParameters
{
    self.outputResults       = nil;
    
    self.trainedNetwork      = nil;
    self.trainingGeneration  = 0;
    
    self._hiddenOutputs      = nil;
    self._outputErrors       = nil;
    self._forceStopTraining  = NO;
    self._isDoneSave         = NO;
    self._patternIndex       = 0;
}

-(void)_initWithVars
{
    self.delegate            = nil;
    self.inputs              = [[NSMutableArray alloc] initWithCapacity:0];
    self.inputWeights        = [[NSMutableArray alloc] initWithCapacity:0];
    self.allHiddenWeights    = [[NSMutableArray alloc] initWithCapacity:0];
    self.allHiddenBiases     = [[NSMutableArray alloc] initWithCapacity:0];
    
    self.hiddenLayers        = [NSMutableArray new];
    
    self.countHiddenLayers   = 0;
    self.outputBias          = 0.1f;
    self.outputGoals         = [NSMutableArray new];
    self.learningRate        = 0.8f;
    self.convergenceError    = 0.001f;
    self.fOfAlpha            = 1;
    self.limitGeneration     = 0;
    self.isTraining          = NO;
    self.trainedInfo         = nil;
    
    self.trainingCompletion  = nil;
    self.eachGeneration      = nil;
    
    [self _resetTrainedParameters];
    
    self._maxMultiple        = 1;
    self._goalValue          = 1.0f;
    self._originalParameters = [NSMutableDictionary new];
}

@end

@implementation KRBPN (fixMethods)

-(void)_stopTraining
{
    self.isTraining = NO;
    /*
    if( self.trainingCompletion )
    {
        self.trainingCompletion(NO, nil);
    }
     */
}

-(void)_completedTraining
{
    self.isTraining  = NO;
    if( self._isDoneSave )
    {
        self._isDoneSave = NO;
        [self saveTrainedNetwork];
    }
    
    if( self.delegate )
    {
        if( [self.delegate respondsToSelector:@selector(krBPNDidTrainFinished:trainedInfo:totalTimes:)] )
        {
            [self.delegate krBPNDidTrainFinished:self trainedInfo:self.trainedInfo totalTimes:self.trainingGeneration];
        }
    }
    
    if( self.trainingCompletion )
    {
        self.trainingCompletion(YES, self.trainedInfo, self.trainingGeneration);
    }
}

-(void)_printEachGeneration
{
    if( self.delegate )
    {
        if( [self.delegate respondsToSelector:@selector(krBPNEachGeneration:trainedInfo:times:)] )
        {
            [self.delegate krBPNEachGeneration:self trainedInfo:self.trainedInfo times:self.trainingGeneration];
        }
    }
    
    if( self.eachGeneration )
    {
        self.eachGeneration(self.trainingGeneration, self.trainedInfo);
    }
}

-(void)_copyParametersToTemporary
{
    if( !self._originalParameters )
    {
        self._originalParameters = [NSMutableDictionary new];
    }
    else
    {
        [self._originalParameters removeAllObjects];
    }
    NSMutableDictionary *_originals = self._originalParameters;
    [_originals setObject:[self.inputs copy] forKey:_kOriginalInputs];
    [_originals setObject:[self.inputWeights copy] forKey:_kOriginalInputWeights];
    [_originals setObject:[self.hiddenLayers copy] forKey:_kOriginalHiddenLayers];
    [_originals setObject:[self.allHiddenWeights copy] forKey:_kOriginalAllHiddenWeights];
    [_originals setObject:[self.allHiddenBiases copy] forKey:_kOriginalAllHiddenBiases];
    [_originals setObject:[NSNumber numberWithDouble:self.outputBias] forKey:_kOriginalOutputBias];
    [_originals setObject:[self.outputGoals copy] forKey:_kOriginalOutputGoals];
    [_originals setObject:[NSNumber numberWithFloat:self.learningRate] forKey:_kOriginalLearningRate];
    [_originals setObject:[NSNumber numberWithDouble:self.convergenceError] forKey:_kOriginalConvergenceError];
    [_originals setObject:[NSNumber numberWithFloat:self.fOfAlpha] forKey:_kOriginalFOfAlpha];
    [_originals setObject:[NSNumber numberWithInteger:self.limitGeneration] forKey:_kOriginalLimitGenerations];
    //[_originals setObject:[NSNumber numberWithInteger:self._maxMultiple] forKey:_kOriginalMaxMultiple];
}

-(void)_recoverOriginalParameters
{
    NSMutableDictionary *_originals = self._originalParameters;
    if( _originals )
    {
        if( [_originals count] > 0 )
        {
            [self.inputs removeAllObjects];
            [self.inputs addObjectsFromArray:[_originals objectForKey:_kOriginalInputs]];
            
            [self.inputWeights removeAllObjects];
            [self.inputWeights addObjectsFromArray:[_originals objectForKey:_kOriginalInputWeights]];
            
            [self.hiddenLayers removeAllObjects];
            [self.hiddenLayers addObjectsFromArray:[_originals objectForKey:_kOriginalHiddenLayers]];
            
            [self.allHiddenWeights removeAllObjects];
            [self.allHiddenWeights addObjectsFromArray:[_originals objectForKey:_kOriginalAllHiddenWeights]];
            
            [self.allHiddenBiases removeAllObjects];
            [self.allHiddenBiases addObjectsFromArray:[_originals objectForKey:_kOriginalAllHiddenBiases]];
            
            [self.outputGoals removeAllObjects];
            [self.outputGoals addObjectsFromArray:[_originals objectForKey:_kOriginalOutputGoals]];
            
            self.outputBias       = [[_originals objectForKey:_kOriginalOutputBias] doubleValue];
            self.learningRate     = [[_originals objectForKey:_kOriginalLearningRate] floatValue];
            self.convergenceError = [[_originals objectForKey:_kOriginalConvergenceError] doubleValue];
            self.fOfAlpha         = [[_originals objectForKey:_kOriginalFOfAlpha] floatValue];
            self.limitGeneration  = [[_originals objectForKey:_kOriginalLimitGenerations] integerValue];
            
            //self._maxMultiple     = [[_originals objectForKey:_kOriginalMaxMultiple] integerValue];
        }
    }
}

/*
 * @ 亂數給指定範圍內的值
 *   - ex : 1.0 ~ -1.0
 */
-(double)_randomMax:(double)_maxValue min:(double)_minValue
{
    /*
     * @ 2014.12.28 PM 20:15
     * @ Noted
     *   - rand() not fits to use here.
     *   - arc4random() fits here, it's the real random number.
     */
    //srand((int)time(NULL));
    //double _random = ((double)rand() / RAND_MAX) * (_maxValue - _minValue) + _minValue;
    //RAND_MAX 是 0x7fffffff (2147483647)，而 arc4random() 返回的最大值则是 0x100000000 (4294967296)，故 * 2.0f 待除，或使用自訂義 ARC4RANDOM_MAX      0x100000000
    double _random = ((double)arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
    //NSLog(@"_random : %lf", _random);
    return _random;
}

-(NSInteger)_formatHiddenNetCount:(NSInteger)_netCount
{
    NSInteger _hiddenNetCount = _netCount;
    //隱藏層神經元 MAX is 20 or device will overloading.
    if( _hiddenNetCount > MAX_HIDDEN_NETS_COUNT )
    {
        _hiddenNetCount = MAX_HIDDEN_NETS_COUNT;
    }
    else if( _hiddenNetCount < 1 )
    {
        //最少 1 顆
        _hiddenNetCount = 1;
    }
    return _hiddenNetCount;
}

/*
 * @ 先針對 Output Goals 輸出期望值集合做資料正規化
 *   - 以免因為少設定映射的期望結果而 Crash
 */
-(void)_formatOutputGoals
{
    NSInteger _goalCount  = [self.outputGoals count];
    NSInteger _inputCount = [self.inputs count];
    //輸出期望值組數 < 輸入向量的 Pattern 組數
    if( _goalCount < _inputCount )
    {
        //將缺少的部份用 0.0f 補滿
        NSMutableArray *_goals = [[NSMutableArray alloc] initWithArray:self.outputGoals];
        NSInteger _diffCount   = _inputCount - _goalCount;
        for( int i=0; i<_diffCount; i++ )
        {
            [_goals addObject:@0.0f];
        }
        self.outputGoals = [[NSMutableArray alloc] initWithArray:_goals];
        [_goals removeAllObjects];
        _goals = nil;
    }
}

-(NSInteger)_calculateLayerNetCountWithInputNumber:(NSInteger)_inputNetCount outputNumber:(NSInteger)_outputNetCount
{
    /*
     * @ 輸入層到隱藏層的輸入層 Net 數量公式
     *   - 1. (輸入層的 Net 數 * 輸出層 Net 數) ^ 1/2
     *     - 本式在多輸入，單輸出時的表現最好，會逐層收斂。
     *     - 本式在多輸入，多輸出時，會發生隱藏層神經元個數會逐層增加或不遞減的問題，得慎用。
     *
     *   - 2. (輸入層的 Net 數 + 輸出層 Net 數) / 2
     *     - 本式在多輸入，多輸出時的表現最好，但也會有可能出現神經元個數不逐層遞減的問題，須注意評估。
     *
     */
    NSInteger _netCount = (NSInteger)powf(( _inputNetCount * _outputNetCount ), 0.5f);
    return [self _formatHiddenNetCount:_netCount];
}

/*
 * @ 將每一層隱藏層的 Nets 偏權值與權重值做初始化分割至不同 Array 的動作
 */
-(void)_initHiddenLayerBiasesAndWeights
{
    NSArray *_hiddenLayers = (NSArray *)self.hiddenLayers;
    if( nil != _hiddenLayers && [_hiddenLayers count] > 0 )
    {
        //取出每一層的隱藏層
        int _layerNumber   = -1;
        for( NSArray *_eachHiddenLayers in _hiddenLayers )
        {
            /*
                _hiddenLayers = [NSMutableArray arrayWithObjects:
             
                 //Hidden Layer 1
                 (NSMutableArray *)@[
                                     //Net 4, @[隱藏層神經元 Net 4 的偏權值], @[隱藏層神經元 Net 4 到下一層神經元的權重值]
                                     (NSMutableArray *)@[@[@-0.4], @[@-0.3, @0.2]],
                                     //Net 5
                                     (NSMutableArray *)@[@[@0.2], @[@-0.2, @0.5]]],
                 
                 //Hidden Layer 2
                 (NSMutableArray *)@[
                                     //Net 6
                                     (NSMutableArray *)@[@[@0.3], @[@-0.5, @0.1]],
                                     //Net 7
                                     (NSMutableArray *)@[@[@0.7], @[@0.2, @0.4]]],
                 
                 nil];
            */
            //是第幾層的隱藏層
            ++_layerNumber;
            //隱藏層每一顆 Net 的偏權值
            NSMutableArray *_hiddenBiases = [NSMutableArray new];
            //輸入向量數據到指定隱藏層間的權重值(線)
            NSMutableArray *_inputWeights = [NSMutableArray new];
            //列舉每一層隱藏層裡的所有 Nets Infomation
            for( NSArray *_eachNets in _eachHiddenLayers )
            {
                for( NSArray *_netSettings in _eachNets )
                {
                    //取出 Hidden Net 的偏權值 (型態是 NSArray)
                    [_hiddenBiases addObject:[_netSettings firstObject]];
                    
                    //取出 Hidden Net 的 Input to next Net 的權重值 (型態是 NSArray)
                    [_inputWeights addObject:[_netSettings lastObject]];
                }
            }
            
            /*
             * @ 以下都會變成 3 層陣列
                 allHiddenBiases = @[
                    //Hidden Layer 1
                    @[
                        //Net 4
                        @[@0.3],
                        //Net 5
                        @[@0.2]
                    ],
                    //Hidden Layer 2 ...
                    //Same as Layer 1
                 ];
             
                 allHiddenWeights = @[
                    //Hidden Layer 1
                    @[
                        //Net 4
                        @[@-0.3, @0.2],
                        //Net 5
                        @[@-0.2, @0.5]
                    ],
                    //Hidden Layer 2 ...
                    //Same as Layer 1
                ];
             */
            
            //新增每一層 Hidden Layer 的所有 Nets 的偏權值
            [self.allHiddenBiases addObject:_hiddenBiases];
            
            //新增每一層 Hidden Layer 的所有 Nets 的 Input to next Nets 的權重值
            [self.allHiddenWeights addObject:_inputWeights];
        }
    }
}

@end

@implementation KRBPN (fixFOfNets)

/*
 * @ 活化函式是 LMS (最徒坡降法)
 */
-(float)_useLMSFOfNetViaSumOfNet:(float)_sumOfNet
{
    return ( 1 / ( 1 + powf(M_E, (-(self.fOfAlpha) * _sumOfNet)) ) );
}

//Waiting for next time for Fuzzy combined.
-(float)_useFuzzyFOfnetViaSumOfNet:(float)_sumOfNet
{
    //Do Fuzzy ...
    return -0.1f;
}

@end

@implementation KRBPN (fixTrainings)

/*
 * @ 計算隱藏層各個神經元( Nets )的輸出值
 *
 *   - 1. 累加神經元的值 SUM(net)
 *     - net(j) = ( SUM W(ji) * X(i) ) + b(j)
 *
 *   - 2. 代入活化函式 f(net), LMS 最小均方法
 *
 */
-(NSArray *)_doHiddenLayerSumResultsFromPreviousLayerInputs:(NSArray *)_layerInputs inputWeights:(NSArray *)_inputWeights netBiases:(NSArray *)_biases
{
    if( !_layerInputs )
    {
        return nil;
    }
    
    //運算完成的 Nets
    //NSMutableArray *_nets = [NSMutableArray new];
    //net(j)
    NSMutableArray *_fOfNets = [NSMutableArray new];
    //輸入層要做 SUM 就必須取出同一維度的值做運算

    //NSLog(@"_layerInputs : %@", _layerInputs);
    //NSLog(@"_inputWeights : %@", _inputWeights);
    //NSLog(@"_biases : %@", _biases);
    
    //以 Hidden Layer 的神經元為維度來源
    //取出有多少維度
    int _totalDimesion       = [_biases count];
    
    //NSLog(@"_totalDimesion : %i\n\n\n", _totalDimesion);
    
    //直接用維度取值作 SUM
    for( int _dimesion = 0; _dimesion < _totalDimesion; _dimesion++ )
    {
        NSArray *_netBiases = [_biases objectAtIndex:_dimesion];
        NSNumber *_netBias  = [_netBiases firstObject];
        
        //NSLog(@"===========================\n\n");
        //NSLog(@"_netBias : %@", _netBias);
        
        //再以同維度做 SUM 方法
        float _sumOfNet = 0;
        /*
         * @ inputs = @[//X1
         *              @[@1, @2, @-1],
         *              //X2
         *              @[@0, @1, @1],
         *              //X3
         *              @[@1, @1, @-2]];
         */
        //有幾個 Input 就有幾個 Weight
        //取出每一個輸入值( Ex : X1 轉置矩陣後的輸入向量 [1, 2, -1] )
        int _inputIndex = -1;
        for( NSNumber *_xi in _layerInputs )
        {
            //NSLog(@"_input : %@", _xi);
            
            ++_inputIndex;
            /*
             * @ 輸入層各向量陣列(值)到隱藏層神經元的權重
             *
             *   inputWeights = @[//W15, W16
             *                    @[@0.2, @-0.3],
             *                    //W25, W26
             *                    @[@0.4, @0.1],
             *                    //W35, W36
             *                    @[@-0.5, @0.2],
             *                    //W45, W46
             *                    @[@-0.1, @0.3]];
             */
            //取出每一個同維度的輸入層到隱藏層的權重
            NSArray *_everyWeights = [_inputWeights objectAtIndex:_inputIndex];
            //將值與權重相乘後累加，例如 : SUM( w15 x 0.2 + w25 x 0.4 + w35 x -0.5 ), SUM( w16 x -0.3 + w26 x 0.1 + w37 x 0.2 ) ...
            float _weight          = [[_everyWeights objectAtIndex:_dimesion] floatValue];
            _sumOfNet             += [_xi floatValue] * _weight;
            
            //NSLog(@"xValue : %f, _weight : %f", [_xi floatValue], _weight);
        }
        
        /*
         * @ 隱藏層神經元的偏權值
         *
         *   - _biases = @[@-0.4, @0.2];
         *
         * @ 作 SUM 加總的本意
         *
         *   - 要明白 SUM 的本意，是指「加總融合」所有輸入向量陣列裡的值 ( 做線性代數的一維矩陣相乘 )
         */
        //加上同維度的隱藏層神經元偏權值
        _sumOfNet    += [_netBias floatValue];
        //代入 LMS 活化函式 ( 用 1 除之，則預設限定範圍在 ~ 1.0 以下 )
        //float _fOfNet = 1 / ( 1 + powf(M_E, (-(self.fOfAlpha) * _sumOfNet)) );
        float _fOfNet = [self _useLMSFOfNetViaSumOfNet:_sumOfNet];
        //加入計算好的輸入向量值，輸入向量是多少維度，輸出就多少維度，例如 : x1[1, 2, 3]，則 net(j) 就要為 [4, 5, 6] 同等維度。( 這似乎有誤，尚未搞懂 囧 )
        [_fOfNets addObject:[NSNumber numberWithFloat:_fOfNet]];
        
        //NSLog(@"Output : %f", _fOfNet);
        //NSLog(@"=========================================\n\n\n");
    }
    
    //@[_fOfNet(4), _fOfNet(5), ...]
    return ( [_fOfNets count] > 0 ) ? (NSArray *)_fOfNets : nil;
}

/*
 * @ 所有隱藏層的權重計算都在這裡完成，在這裡不斷遞迴運算每一個隱藏層
 */
-(NSArray *)_sumHiddenLayerNetWeightsFromInputs:(NSArray *)_inputs
{
    //運算完成的 Nets
    NSMutableArray *_toOutputNets = [NSMutableArray new];
    NSArray *_hiddenLayers        = (NSArray *)self.hiddenLayers;
    
    //NSLog(@"_hiddenLayers : %@", _hiddenLayers);
    //NSLog(@"allHiddenWeights : %@", self.allHiddenWeights);
    //NSLog(@"allHiddenBiases : %@", self.allHiddenBiases);
    
    int _layerCount               = [_hiddenLayers count];
    if( nil != _hiddenLayers && _layerCount > 0 )
    {
        NSArray *_nextInputs  = nil;
        //列舉每一層的隱藏層
        for( int _layerNumber = 0; _layerNumber < _layerCount; _layerNumber++ )
        {
            //要做計算的輸入向量數據
            NSArray *_layerInputs  = nil;
            NSArray *_inputWeights = nil;
            NSArray *_netBiases    = nil;
            //是第 1 層 Hidden Layer，就直接做原始的 Input Layer 到 Hidden Layer 的處理
            if( _layerNumber < 1 )
            {
                _layerInputs  = _inputs;
                _inputWeights = self.inputWeights;
                _netBiases    = [self.allHiddenBiases objectAtIndex:_layerNumber];
            }
            else
            {
                //是第 2 層以後的 Hidden Layer，就做上一層 Hidden Layer 輸出的 Outputs 當成下一層 Hidden Layer 的 Inputs 處理
                _layerInputs  = [_toOutputNets lastObject];
                //取出上一層指到本隱藏層的線權重( ex : net 5 -> net 7 )
                _inputWeights = [self.allHiddenWeights objectAtIndex:_layerNumber - 1];
                _netBiases    = [self.allHiddenBiases objectAtIndex:_layerNumber];
            }
            
            //要做為下一層隱藏層的 Inputs
            _nextInputs = [self _doHiddenLayerSumResultsFromPreviousLayerInputs:_layerInputs
                                                                   inputWeights:_inputWeights
                                                                      netBiases:_netBiases];
            
            //存入每一個隱藏層的 Output 值
            if( nil != _nextInputs )
            {
                //2 層陣列，依照 Hidden Layer # 存放每一層 Hidden Layer 裡每一顆 Net 作完 SUM 後的所有值
                //Hidden Layer 1, Hidden Layer 2, ...
                //@[ @[0.1, 0.2], @[-0.2, -0.3, 0.5], ... ]
                [_toOutputNets addObject:[_nextInputs copy]];
            }
            
            _nextInputs = nil;
        }
    }
    
    //NSLog(@"_toOutputNets : %@", _toOutputNets);
    
    //這裡最後要 return 的是「所有 Hidden Layer 一路到 Output Layer 的輸出結果」
    return ( [_toOutputNets count] > 0 ) ? (NSArray *)_toOutputNets : nil;
}

/*
 * @ 計算輸出層神經元的值 ( 最後的輸出結果 )
 *   - 跟計算隱藏層 [self _doHiddenLayerSumResultsFromPreviousLayerInputs:::] 一樣的公式。
 */
-(NSArray *)_sumOutputLayerNetsValue
{
    NSMutableArray *_nets    = [NSMutableArray new];
    //取出最後一層 Hidden Layer 的 Output 值
    NSArray *_lastHiddenNets = (NSArray *)[self._hiddenOutputs lastObject];
    if( _lastHiddenNets )
    {
        NSMutableArray *_fOfNets = [NSMutableArray new];
        float _sumOfNet          = 0;
        /*
         * @ 最後一層隱藏層神經元到輸出層神經元的權重值
         *   
         *   hiddenNetWeights = @[//W46
         *                        @-0.3,
         *                        //W56
         *                        @-0.2];
         */
        /*
            _lastHiddenLayers = @[
              //Net 7
              @[@0.3],
              //Net 8
              @[@0.2],
              //Net 9
              @[@-0.1]
              ],
            ];
            
            _lastHiddenNets = @[-0.2, -0.3, 0.5];
        */
        //取得最後一層 Hidden Layer 到 Output Layer 的 Weights
        NSArray *_lastHiddenLayers = [self.allHiddenWeights lastObject];
        int _sameIndex             = -1;
        for( NSNumber *_netValue in _lastHiddenNets )
        {
            ++_sameIndex;
            NSArray *_hiddenWeights = [_lastHiddenLayers objectAtIndex:_sameIndex];
            for( NSNumber *_everyWeights in _hiddenWeights )
            {
                float _xValue = [_netValue floatValue];
                float _weight = [_everyWeights floatValue];
                _sumOfNet    += _xValue * _weight;
                //NSLog(@"xValue : %f, _weight : %f\n\n\n", _xValue, _weight);
            }
        }
        _sumOfNet    += self.outputBias;
        float _fOfNet = [self _useLMSFOfNetViaSumOfNet:_sumOfNet];
        [_fOfNets addObject:[NSNumber numberWithFloat:_fOfNet]];
        [_nets addObject:_fOfNets];
        
        //@[@[_fOfNets(output net)]]
        return _nets;
    }
    return nil;
}

/*
 * @ 計算輸出層神經元( Net )的輸出誤差 ( 與期望值的誤差 )
 *   - 公式 : Oj x ( 1 - Oj ) x ( Tj - Oj )
 */
-(NSArray *)_calculateOutputError
{
    self.outputResults = [self _sumOutputLayerNetsValue];
    NSArray *_nets     = self.outputResults;
    //NSLog(@"output net values : %@", _nets);
    if( _nets )
    {
        NSMutableArray *_errors = [NSMutableArray new];
        for( NSArray *_outputs in _nets )
        {
            for( NSNumber *_output in _outputs )
            {
                //取出輸出層神經元的輸出結果
                float _outputValue = [_output floatValue];
                //計算與期望值的誤差
                float _errorValue  = _outputValue * ( 1 - _outputValue ) * ( self._goalValue - _outputValue );
                [_errors addObject:[NSNumber numberWithFloat:_errorValue]];
            }
        }
        //NSLog(@"_errors : %@", _errors);
        
        //@[Output Error Value]
        return _errors;
    }
    return nil;
}

/*
 * @ 計算所有隱藏層神經元( Nets )的誤差
 *   - 公式 : Oj x ( 1 - Oj ) x Errork x Wjk
 *
 *   Input Layer   Hidden Layer 1   Hidden Layer 2   Hidden Layer 3   Output Layer
 *
 *     Net 1
 *     Net 2         Net 5            Net 7             Net 9           Net 11
 *     Net 3         Net 6            Net 8             Net 10
 *     Net 4
 *
 */
-(NSArray *)_calculateNetsError
{
    //取得輸出層輸出誤差 ( 與期望值的誤差 )
    self._outputErrors        = [self _calculateOutputError];
    NSArray *_netExpectErrors = self._outputErrors;
    
    //NSLog(@"allHiddenWeights : %@", self.allHiddenWeights);
    //NSLog(@"allHiddenBiases : %@", self.allHiddenBiases);
    
    if( _netExpectErrors )
    {
        //NSMutableArray *_netErrors = [NSMutableArray new];
        
        NSMutableArray *_allErrors = nil;
        
        //Hidden Layer 1, Hidden Layer 2, ...
        //Ex : @[ @[0.1, 0.2], @[-0.2, -0.3, 0.5], ... ]
        NSArray *_allHiddenOutputs = self._hiddenOutputs;
        
        int _hiddenCount           = [_allHiddenOutputs count];
        
        //NSLog(@"_allHiddenOutputs : %@", _allHiddenOutputs);
        //NSLog(@"_hiddenCount : %i", _hiddenCount);
        
        //取出輸出層的誤差值倒算回去每一個神經元的誤差值 ( 1 個 Output Net 就只會有 1 筆資料，多 Output 才會有多筆資料 )
        for( NSNumber *_outpurError in _netExpectErrors )
        {
            /*
             * @ 第 1 次依 Output Net 11 反算回去 Net 9, Net 10 的 Error Value
             * @ 第 2 次是依 Net 9, Net 10 的 Error Value 當成 Input 反算回去 Net 7, Net 8 的 Error Value
             *   - 在遇到同樣是隱藏層的 Nets 時，因為會有同一個 Net 有多個 Error Value 的關係，而產生問題，例如 :
             *     - Net 7 對應 Net 5 與 Net 6，則 Net 7 和 Net 8 在反算回去時，會各自產生 1 組 Net75 與 Net85 的 Error Value，
             *       Net 5 就會有 2 組 Net Error Values。
             *     - 由於 Net 5 在正傳遞時，其 Net Output 值會讓 Net 7 與 Net 8 各自做 SUM 加總，故同理，則反回用 Net75 與 Net85 的 Error Values，
             *       讓 Net 5 做這 2 組 Net Error Values 的「SUM 加總」，當成 Net 5 的 Error Value 即可。
             *
             *   - 未來要 Enhance 的話，就能在這裡使用不同的演算方法，來替代 SUM + LMS 求出共同誤差值 ( 例如求出 Net 5 的共同誤差值 )。
             *
             * @ 依附 Sample Network 所示，現在求出 Net 11 的期望誤差後，反算回去修正 Net 9, Net 10 的 Bias 是 OK 的，因為是 1 對 1 的關聯性。
             *   而反求回 Hidden Layer 2 的 Net 7, 8 就會遇到 Data Fusion 的問題，Net 9, 10 的誤差值是由 Net 11 所計算取得，
             *   此時 Net 7 照理說就會擁有 2 組由 Net 9, 10 所各自算出來的誤差值。那麼，Net 7 就會遇到 2 組誤差值同時產生的問題，
             *   而必須再將 2 組誤差值融合成單一誤差值，才能針對 Net 7 修正 Bias。加權值都是 1 對 1，無此問題。但 Net Bias 則是 1 對多，而會有此問題。
             *   因為不想針對 Net 7 再做 1 次 SUM + LMS，這會造成局部收斂解。
             *
             * @ " 待找出有什麼作法可以解決修正 Net 5, 6, 7, 8 Biases 的最佳解法 ? "
             *
             */
            
            /*
             * @ 取出每一顆神經元
             * @ 並計算該隱藏層的誤差值
             *   - 遍歷該隱藏層裡相對應的所有神經元
             *   - 第 1 次會得到 Net 9, Net 10 的 Error Values
             *   - 第 2 次要使用 Net 7, Net 8 的 Error Values 當 Input 計算 Net 5, Net 6 的 Error Values
             *   - 其餘隱藏層都以此類推反算
             */
            
            //倒取每一層隱藏層的 Output 結果
            for( int _layerIndex = _hiddenCount - 1; _layerIndex >= 0; _layerIndex-- )
            {
                //Current Hidden Layer errors
                NSMutableArray *_layerErrors = [NSMutableArray new];
                
                //第 1 次要先算最後 1 層 Hidden Layer error value
                if( nil == _allErrors )
                {
                    //Put the net 9 output error
                    //先統一將 Output Layer 的 Error 加入至 _allErrors 裡進行統一的流程運算(如此便不用重複寫 Code)，之後 return 時再將第 1 個 Object 刪除即可
                    [_layerErrors addObject:_outpurError];
                    _allErrors = [NSMutableArray new];
                    //這樣裡面的 _layerErrors 物件才不會跟外部和下一次循環增加物件時，因為記憶體弱連結的關係而連動到，造成 Exceptions.
                    [_allErrors addObject:[_layerErrors copy]];
                    [_layerErrors removeAllObjects];
                }
                
                //取出倒算的當前 Hidden Layer 的 SUM output results
                NSArray *_layerOutputs = [_allHiddenOutputs objectAtIndex:_layerIndex];
                
                //NSLog(@"第 %i 層 Hidden Layer", _layerIndex);
                //NSLog(@"_layerOutputs : %@", _layerOutputs);
                
                /*
                 * @ 順序
                 *   - Net 11 error -> Net 9, 10 error -> Net 7, 8 error -> Net 5, 6 error
                 * 
                 * @ 依照要推算回去的上一層 Nets 來做交叉 SUM 演算
                 *   - 拿上一層計算好的隱藏層誤差當成本隱藏層要修正的 Output Error，
                 *   對其餘 Hidden Layer 做 SUM，參考了文獻且考慮過後，統一不做 LMS 計算最小誤差，以避免誤差值過度縮小而產生異常。
                 *
                 */
                //取出 PreviousNets 與依照其維度進行取值運算
                //NSArray *_previousNets = [self.allHiddenBiases objectAtIndex:_layerIndex];
                int _whichNet          = -1;
                int _netsCount         = [_layerOutputs count]; //[_previousNets count]; //Outputs Count 跟 Previous Nets Count 是相等的
                for( int _dimesion = 0; _dimesion < _netsCount; _dimesion++ )
                {
                    ++_whichNet;
                    //是哪一顆 Net 的 Output 值要做 SUM，例如，計算 Net 7 時，這 _whichWeight 指的是 Net 7 連結 Net 9 和 Net 10 的哪幾條 Weight
                    NSArray *_netWeights = [[self.allHiddenWeights objectAtIndex:_layerIndex] objectAtIndex:_whichNet];
                    //要取出哪一個 Output Value
                    int _outputIndex     = -1;
                    //要取出哪一個 Error Value
                    int _errorIndex      = -1;
                    
                    /*
                    //代表只有 1 條權重值 ( 最後的輸出層為單一 Output ) ; Error Value 沒有此限制
                    if( [_netWeights count] == 1 )
                    {
                        //如此便能做到 1 對 1 的取出正確相對應位置的 Output 值
                        _outputIndex     = _whichNet - 1;
                    }
                     */
                    
                    //針對每一個 Net 的誤差值做 SUM
                    float _sumError         = 0.0f;
                    NSArray *_lastErrors    = [_allErrors lastObject];
                    _outputIndex            = _dimesion;
                    NSNumber *_netOutput    = [_layerOutputs objectAtIndex:_outputIndex];
                    
                    //NSArray *_netBiases     = [_previousNets objectAtIndex:_dimesion];
                    //NSLog(@"_netBiases : %@", _netBiases);
                    //NSLog(@"_layerOutputs : %@", _layerOutputs);
                    //NSLog(@"_lastErrors : %@", _lastErrors);
                    //NSLog(@"_netWeights : %@", _netWeights);
                    
                    for( NSNumber *_netWeight in _netWeights )
                    {
                        //++_outputIndex;
                        ++_errorIndex;
                        
                        //NSLog(@"_outputIndex : %i", _outputIndex);
                        //NSLog(@"_errorIndex : %i", _errorIndex);
                        
                        //NSNumber *_netOutput = [_layerOutputs objectAtIndex:_outputIndex];
                        NSNumber *_lastError = [_lastErrors objectAtIndex:_errorIndex];
                        
                        //Output7 x ( 1 - Output7 ) x Error9 x W79
                        //0.332   x ( 1 - 0.332 )   x 0.1311 x -0.3
                        //作 SUM 把 Input 進來有相映射關聯的 Net Errors 都加起來
                        float _expectError   = [_lastError floatValue];
                        float _weight        = [_netWeight floatValue];
                        float _output        = [_netOutput floatValue];
                        _sumError           += _output * ( 1 - _output ) * _expectError * _weight;
                        
                        //NSLog(@"_output, _expectError, _weight : %@, %@, %@", _netOutput, _lastError, _netWeight);
                    }
                    
                    //NSLog(@"_sumError : %f", _sumError);
                    //NSLog(@"===========================================\n\n\n");
                    
                    //將本層算出來的誤差值，當成上一層 hidden layer 的誤差值使用，for ex : 第 1 次運算後會有 2 筆資料
                    [_layerErrors addObject:[NSNumber numberWithFloat:_sumError]];
                    
                }
                
                [_allErrors addObject:_layerErrors];

            }
            
            //刪除第 1 筆僅用於統一運算的權宜資料 ( Output Layer Error )，以恢復正常資料列
            //[_allErrors removeObjectAtIndex:0];
            
            //多陣列內容的用意是預留之後要做多輸入多輸出的回饋修正的擴充彈性
            //[_netErrors addObject:_allErrors];
        }
        
        //@[ Hidden Layer 2 errors, Hidden Layer 1 errors ];
        //@[ @[ @[Net 7 error, Net 8 error], @[Net 4 error, Net 5 error, Net 6 error], ... ] ]
        return _allErrors;
        //return _netErrors;
    }
    return nil;
}

/*
 * @ 更新權重與偏權值
 *   - 公式 : Shita(i) = Shita(i) + learning rate x Error(k)
 *              偏權值 = 偏權值    +   學習速率      x 要修改的誤差值
 */
-(BOOL)_refreshNetsWeights
{
    if( self._forceStopTraining )
    {
        [self _stopTraining];
        return NO;
    }
    
    self.isTraining = YES;
    BOOL _onePatternTrained = NO;
    //隱藏層神經元的輸出誤差值
    //@[ @[Net 9 error], @[Net 7 error, Net 8 error], @[Net 4 error, Net 5 error, Net 6 error], ... ]
    NSArray *_netErrors  = [self _calculateNetsError];
    //NSLog(@"_netErrors : %@\n\n\n", _netErrors);
    
    if( _netErrors )
    {
        /*
         * @ 列舉所有的 Output Errors
         *   - 1. 先更新輸出層神經元的偏權值
         *   - 2. 再更新所有隱藏層神經元的權重與偏權值
         */
        int _layerIndex       = -1;
        int _hiddenLayerIndex = [self.allHiddenWeights count]; //取出共有幾層 Hidden Layer
        
        //NSLog(@"_hiddenLayerIndex : %i", _hiddenLayerIndex);
        //NSLog(@"self.allHiddenWeights : %@", self.allHiddenWeights);
        
        //NSLog(@"========================================== \n\n\n");
        
        if( _hiddenLayerIndex < 1 )
        {
            //TODO : 想想怎麼 enhance performance 和防呆，當 _hiddenLayerIndex = 0 時
            return NO;
        }
        
        //Copy arraies to avoid the memory synchorized reference problems when use [self.allHiddenWeights replaceObjectAtIndex::];
        //NSMutableArray *_tempHiddenWeights = [self.allHiddenWeights copy];
        //NSMutableArray *_tempHiddenBiases  = [self.allHiddenBiases copy];
        
        //NSLog(@"_tempHiddenWeights : %@", _tempHiddenWeights);
        //NSLog(@"_tempHiddenBiases : %@", _tempHiddenBiases);
        //NSLog(@"_allHiddenLayerOutputs : %@", self._hiddenOutputs);
        
        //有幾層 Layer 就跑幾次
        for( NSArray *_eachErrors in _netErrors )
        {
            ++_layerIndex;
            --_hiddenLayerIndex;
            
            
            //NSLog(@"_hiddenLayerIndex : %i", _hiddenLayerIndex);
            //NSLog(@"_eachErrors : %@", _eachErrors);
            
            if( _hiddenLayerIndex < 0 )
            {
                break;
            }
            
            //NSLog(@"_hiddenLayerOutputs : %@", [self._hiddenOutputs objectAtIndex:_hiddenLayerIndex]);
            
            //先更新 Output Layer bias value
            //Is output layer to hidden layer
            if( _layerIndex < 1 )
            {
                for( NSNumber *_outputError in _eachErrors )
                {
                    //Update the net bias of output layer.
                    self.outputBias = self.outputBias + ( self.learningRate * [_outputError floatValue] );
                    //NSLog(@"Updated outputBias to %f", self.outputBias);
                }
            }
            
            //再更新 Hidden Layer 的 Biases 和 Weights
            NSMutableArray *_newLayerWeights = [NSMutableArray new];
            NSMutableArray *_newLayerBiases  = [NSMutableArray new];
            
            NSArray *_previousNets = [self.allHiddenBiases objectAtIndex:_hiddenLayerIndex];
            
            int _whichNet = -1;
            for( NSArray *_netBiases in _previousNets )
            {
                ++_whichNet;
                
                NSMutableArray *_updatedWeights = [NSMutableArray new];
                NSMutableArray *_updatedBiases  = [NSMutableArray new];
                
                for( NSNumber *_netBias in _netBiases )
                {
                    NSArray *_netWeights = [[self.allHiddenWeights objectAtIndex:_hiddenLayerIndex] objectAtIndex:_whichNet];
                    int _dimension = -1;
                    
                    //NSLog(@"_netBiases : %@", _netBiases);
                    //NSLog(@"_netWeights : %@", _netWeights);
                    //NSLog(@"weight count : %i", [_netWeights count]);
                    
                    /*
                     * @ 更新加權值( Weights )與偏權值( Bias )
                     *   - 1. 在計算更新加權值時，同時計算更新偏權值時 error value
                     *        - Error Value 是做 SUM 加總，把有對應到的 Error(k) 的值都做 SUM 加總起來，變成總誤差值才反算回去
                     *   - 2. 下面迴圈在取出該 Net 每一條線的加權值，同時計算偏權值的 Error Value
                     *
                     * @ for example : 
                     *   - Net 7 的 Bias Error value 是由 Net 9, 10 的 error value 加總起來的
                     */
                    float _sumBiasError    = 0.0f;
                    //取出該 Net 每一條線的加權值
                    for( NSNumber *_netWeight in _netWeights )
                    {
                        ++_dimension;
                        float _netError    = [[_eachErrors objectAtIndex:_dimension] floatValue];
                        float _weight      = [_netWeight floatValue];
                        NSArray *_outputs  = [self._hiddenOutputs objectAtIndex:_hiddenLayerIndex];
                        float _netOutput   = [[_outputs objectAtIndex:_whichNet] floatValue];
                        float _resetWeight = _weight + ( self.learningRate * _netError * _netOutput );
                        [_updatedWeights addObject:[NSNumber numberWithFloat:_resetWeight]];
                        
                        _sumBiasError     += _netError;
                        
                        //NSLog(@"update weight, the error : %f, output : %f, weight : %f", _netError, _netOutput, _weight);
                    }
                    
                    //NSLog(@"_whichNet : %i", _whichNet);
                    
                    //_sumBiasError = [self _useFuzzyFOfnetViaSumOfNet:_sumBiasError];
                    
                    //更新偏權值
                    //取出指定的 Net 偏權值
                    //NSLog(@"_netBias : %@, _errorValue : %f", _netBias, _sumBiasError);
                    float _updatedBias = [_netBias floatValue] + ( self.learningRate * _sumBiasError );
                    [_updatedBiases addObject:[NSNumber numberWithFloat:_updatedBias]];
                    
                }
                
                //NSLog(@"_updatedWeights : %@", _updatedWeights);
                //NSLog(@"_updatedBiases : %@", _updatedBiases);
                
                [_newLayerWeights addObject:_updatedWeights];
                [_newLayerBiases addObject:_updatedBiases];
                
                //NSLog(@"====================================================== \n\n");
            }
            
            //NSLog(@"_newLayerWeights : %@", _newLayerWeights);
            //NSLog(@"_newLayerBiases : %@", _newLayerBiases);
            
            //直接覆蓋修正後的加權值 (Performance 比一個一個覆蓋子元素的好)
            [self.allHiddenWeights replaceObjectAtIndex:_hiddenLayerIndex withObject:_newLayerWeights];
            
            //直接覆蓋修正後的隱藏層偏權值
            [self.allHiddenBiases replaceObjectAtIndex:_hiddenLayerIndex withObject:_newLayerBiases];
            
        }
        
        //NSLog(@"allHiddenWeights : %@", self.allHiddenWeights);
        //NSLog(@"allHiddenBiases : %@", self.allHiddenBiases);
        
        /*
         * @ 最後更新所有輸入層到第一層隱藏層的權重
         *   - //@[ @[W14, W15, W16], @[W24, W25, W26], @[W34, W35, @36] ]
         *     _inputWeights = @[ @[@0.2, @-0.3], @[@0.4, @0.1], @[@-0.5, @0.2] ]
         *
         * @ 不跟上面的迴圈合在一起做的原因，是為了可讀性與未來 Enhance 時的彈性
         */
        NSArray *_weights = [self.inputWeights copy];
        int _inputIndex   = -1;

        //取出第一層的 Hidden Layer errors
        //@[Net 4 error, Net 5 error, Net 6 error]
        NSArray *_errors  = [_netErrors lastObject];
        
        for( NSArray *_netWeights in _weights )
        {
            ++_inputIndex;
            int _weightIndex              = -1;
            NSMutableArray *_resetWeights = [NSMutableArray new];
            for( NSNumber *_everyWeight in _netWeights )
            {
                //每一個加權值(線權重)陣列的元素個數，會等於隱藏層神經元個數
                ++_weightIndex;
                float _netWeight   = [_everyWeight floatValue];
                float _hiddenError = [[_errors objectAtIndex:_weightIndex] floatValue];
                float _inputValue  = [[[self.inputs objectAtIndex:self._patternIndex] objectAtIndex:_inputIndex] floatValue];
                float _resetWeight = _netWeight + ( self.learningRate * _hiddenError * _inputValue );
                [_resetWeights addObject:[NSNumber numberWithFloat:_resetWeight]];
                //NSLog(@"_new weight : %f = %f + ( %f * %f * %f )", _resetWeight, _netWeight, self.learningRate, _hiddenError, _inputValue);
            }
            //修正 InputWeights 輸入層到第一層隱藏層的權重
            [self.inputWeights replaceObjectAtIndex:_inputIndex withObject:_resetWeights];
        }
        
        _weights           = nil;
        _onePatternTrained = YES;
    }
    
    return _onePatternTrained;
}

//公式化訓練用的期望值
-(void)_formatMaxMultiple
{
    //先找出期望值的最大絕對值
    NSNumber *_max  = [self.outputGoals valueForKeyPath:@"@max.self"];
    NSNumber *_min  = [self.outputGoals valueForKeyPath:@"@min.self"];
    double _fabsMax = fabs(_max.doubleValue);
    double _fabsMin = fabs(_min.doubleValue);
    double _realMax = MAX(_fabsMax, _fabsMin);
    if( _realMax > 1.0f )
    {
        self._maxMultiple  = 10 * ( (int)log10(_realMax) + 1 );
    }
}

-(void)_startTraining
{
    ++self.trainingGeneration;
    self._patternIndex = -1;
    /*
     * @ 依公式所說，X(i) 輸入向量應做轉置矩陣運算，但轉置矩陣須耗去多餘效能，
     *   因此，這裡暫不採用直接先轉成轉置矩陣的動作，
     *   而是直接依照資料長度取出同維度的方式來做轉置矩陣。
     *
     * @ 如輸入向量是 X1 = [1, 2, 3]; 的多值型態，就採用線性代數的解法，
     *   - 要將 X1 先轉置矩陣變成 :
     *             [1]
     *     X1(T) = [2]
     *             [3]
     *     這為第 1 筆訓練資料，當成輸入層神經元代入，此時輸入層就有 3 顆神經元。
     */
    //先正規化處理資料，以避免異常狀況發生
    [self _formatOutputGoals];
    
    //開始代入 X1, X2 ... Xn 各組的訓練資料
    for( NSArray *_inputs in self.inputs )
    {
        ++self._patternIndex;
        /*
         * @ 每一筆輸入向量( 每組訓練的 Pattern )都會有自己的輸出期望值
         *   － 例如 : 
         *           輸入 X1[1, 0, 0]，其期望輸出為 1.0
         *           輸入 X2[0, 1, 0]，其期望輸出為 2.0
         *           輸入 X3[0, 0, 1]，其期望輸出為 3.0
         *      以此類推。
         */
        //不論正負號都先轉成絕對值，我只要求得除幾位數變成小數點
        //NSLog(@"%i", (int)log10f(-81355.555)); //-2147483648
        //NSLog(@"%i", (int)log10f(81355.555)); //4 個 10 倍
        self._goalValue     = [[self.outputGoals objectAtIndex:self._patternIndex] doubleValue] / self._maxMultiple;
        self._hiddenOutputs = [self _sumHiddenLayerNetWeightsFromInputs:_inputs];
        //NSLog(@"\n\n_goalValue : %lf, _hiddenOutputs : %@\n\n\n", self._goalValue, self._hiddenOutputs);
        //更新權重失敗，代表訓練異常，中止 !
        if ( ![self _refreshNetsWeights] )
        {
            //考慮，是否要記錄訓練到哪一筆，等等再繼續 ?
            //要繼續的話應該要重頭再來才是 ?
            break;
        }
    }
    
    //如有指定迭代數 && 當前訓練迭代數 >= 指定迭代數
    if( self.limitGeneration > 0 && self.trainingGeneration >= self.limitGeneration )
    {
        //停止訓練
        [self _completedTraining];
        return;
    }
    
    //檢查是否收斂
    BOOL _isGoalError = NO;
    for( NSNumber *_outpurError in self._outputErrors )
    {
        //使用絕對值來做誤差比較
        float _resultError = fabsf( [_outpurError floatValue] );
        //如果已達收斂誤差，就不再繼續訓練
        if( _resultError <= self.convergenceError )
        {
            _isGoalError = YES;
            break;
        }
    }
    
    if( _isGoalError )
    {
        //達到收斂誤差值或出現異常狀況，即停止訓練
        [self _completedTraining];
        return;
    }
    else
    {
        //全部數據都訓練完了，才為 1 迭代
        [self _printEachGeneration];
        //未達收斂誤差，則繼續執行訓練
        [self _startTraining];
    }
}

-(void)_trainingWithExtraSetupHandler:(void(^)())_extraSetupHandler
{
    //DISPATCH_QUEUE_CONCURRENT
    dispatch_queue_t queue = dispatch_queue_create("com.krbpn.train-network", NULL);
    dispatch_async(queue, ^(void)
    {
        [self pause];
        [self _resetTrainedParameters];
        [self _initHiddenLayerBiasesAndWeights];
        if( _extraSetupHandler )
        {
            _extraSetupHandler();
        }
        [self _copyParametersToTemporary];
        dispatch_async(dispatch_get_main_queue(), ^
        {
            [self _formatMaxMultiple];
            [self _startTraining];
        });
    });
}

@end

@implementation KRBPN

@synthesize delegate            = _delegate;

@synthesize inputs              = _inputs;
@synthesize inputWeights        = _inputWeights;
@synthesize allHiddenWeights    = _allHiddenWeights;
@synthesize allHiddenBiases     = _allHiddenBiases;

@synthesize hiddenLayers        = _hiddenLayers;

@synthesize countHiddenLayers   = _countHiddenLayers;
@synthesize countOutputNets     = _countOutputNets;
@synthesize countInputNets      = _countInputNets;

@synthesize outputBias          = _outputBias;
@synthesize outputResults       = _outputResults;
@synthesize outputGoals         = _outputGoals;
@synthesize learningRate        = _learningRate;
@synthesize convergenceError    = _convergenceError;
@synthesize fOfAlpha            = _fOfAlpha;
@synthesize limitGeneration     = _limitGeneration;
@synthesize trainingGeneration  = _trainingGeneration;
@synthesize isTraining          = _isTraining;
@synthesize trainedInfo         = _trainedInfo;
@synthesize trainedNetwork      = _trainedNetwork;

@synthesize trainingCompletion  = _trainingCompletion;
@synthesize eachGeneration      = _eachGeneration;

@synthesize _hiddenOutputs;
@synthesize _goalValue;
@synthesize _outputErrors;
@synthesize _forceStopTraining;
@synthesize _originalParameters;
@synthesize _isDoneSave;
@synthesize _patternIndex;
@synthesize _maxMultiple;

+(instancetype)sharedNetwork
{
    static dispatch_once_t pred;
    static KRBPN *_object = nil;
    dispatch_once(&pred, ^
    {
        _object = [[KRBPN alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        [self _initWithVars];
    }
    return self;
}

#pragma --mark Settings Public Methods
/*
 * @ 各輸入向量陣列值 & 每一筆輸入向量的期望值( 輸出期望 )
 */
-(void)addPatterns:(NSArray *)_patterns outputGoal:(float)_goal
{
    [_inputs addObject:_patterns];
    [_outputGoals addObject:[NSNumber numberWithFloat:_goal]];
}

/*
 * @ 輸入層各向量值到隱藏層神經元的權重
 *   - 連結同一個 Net 的就一組一組分開，有幾個 Hidden Net 就會有幾組
 */
-(void)addPatternWeights:(NSArray *)_weights
{
    //@[ @[W14, W15], @[W24, W25], @[W34, W35] ]
    //@[ @[@0.2, @-0.3], @[@0.4, @0.1], @[@-0.5, @0.2] ]
    [_inputWeights addObject:_weights];
}

/*
 * @ 增加隱藏層的各項參數設定
 *   - _layerIndex 第幾層的 Hidden Layer
 *   - _netBias 隱藏層神經元 Net # 的偏權值
 *   - _netWeights 隱藏層神經元 Net # 到下一層神經元的權重值
 */
-(void)addHiddenLayerAtIndex:(int)_layerIndex netBias:(float)_netBias netWeights:(NSArray *)_netWeights
{
    /*
     * @ Example :
        _hiddenLayers = [NSMutableArray arrayWithObjects:
     
         //Hidden Layer 1
         (NSMutableArray *)@[
           //Net 4, @[隱藏層神經元 Net 4 的偏權值], @[隱藏層神經元 Net 4 到下一層神經元的權重值]
           (NSMutableArray *)@[@[@-0.4], @[@-0.3, @0.2]],
           //Net 5
           (NSMutableArray *)@[@[@0.2], @[@-0.2, @0.5]]],
         
         //Hidden Layer 2
         (NSMutableArray *)@[
           //Net 6
           (NSMutableArray *)@[@[@0.3], @[@-0.5, @0.1]],
           //Net 7
           (NSMutableArray *)@[@[@0.7], @[@0.2, @0.4]]],
     
         nil];
    */
    
    NSMutableArray *_layers = nil;
    NSMutableArray *_nets   = [NSMutableArray new];
    [_nets addObject:@[@[[NSNumber numberWithFloat:_netBias]], _netWeights]];
    
    /*
     * @ 開始進行數據的檢查
     */
    //有指定的 Hidden Layer 層
    if( (_layerIndex + 1) <= [_hiddenLayers count] )
    {
        //進行該 Layer 層的 Add Nets 動作
        //先取出指定的 Hidden Layer
        _layers = [_hiddenLayers objectAtIndex:_layerIndex];
        //新增 Nets
        [_layers addObject:_nets];
        //不用 replace 回去，因為是弱連結，記憶體共用
    }
    else
    {
        //無指定的 Hidden Layer 層，就直接增加 Hidden Layer
        _layers = [[NSMutableArray alloc] initWithObjects:_nets, nil];
        [_hiddenLayers addObject:_layers];
    }
    
    //NSLog(@"_layers : %@", _layers);
    
}

#pragma --mark Setting Paramaters Public Methods
/*
 * @ Get Random Number
 */
-(double)randomMax:(double)_maxValue min:(double)_minValue
{
    return [self _randomMax:_maxValue min:_minValue];
}

/*
 * @ 估算約會有幾層隱藏層
 */
-(NSInteger)evaluateHiddenLayerNumbers
{
    NSInteger _inputNetCount  = self.countInputNets;
    NSInteger _outputNetCount = self.countOutputNets;
    NSInteger _layerNetCount  = [self _calculateLayerNetCountWithInputNumber:_inputNetCount outputNumber:_outputNetCount];
    NSInteger _layerNumber    = 0;
    
    for( int i = 0; i<_layerNetCount; i++ )
    {
        //本層有幾顆
        _layerNetCount     = [self _calculateLayerNetCountWithInputNumber:_inputNetCount outputNumber:_outputNetCount];
        if( _layerNetCount <= _outputNetCount )
        {
            break;
        }
        
        ++_layerNumber;
        
        if( _layerNumber > MAX_HIDDEN_LAYER_COUNT )
        {
            _layerNumber = MAX_HIDDEN_LAYER_COUNT;
            break;
        }
        
        //計算下一層有幾顆
        _inputNetCount   = _layerNetCount;
    }
    
    return _layerNumber;
}

/*
 * @ Random all hidden-net weights, net biases, output net bias.
 *   - 亂數設定隱藏層神經元的權重、神經元偏權值、輸出層神經元偏權值
 *
 * @ 如果不指定神經元的權重值，那就自行依照「輸入向量的神經元個數做平方」，也就是輸入層如 2 個神經元，
 *   則隱藏層神經元就預設為 2^2 = 4 個 ( 參考 ANFIS ) 的作法，而每一個輸入層到隱藏層的權重值，
 *   就直接亂數給 -1.0 ~ 1.0 之間的值，而每一個神經元的偏權值也是亂數給 -1.0 ~ 1.0。
 */
-(void)randomHiddenLayerWeightsWithTotalLayers:(int)_totalLayers
{
    //NSLog(@"_totalLayers : %i", _totalLayers);
    
    //At least for 1 hidden layer.
    if( _totalLayers < 1 )
    {
        _totalLayers = 1;
    }
    
    //先清空舊資料
    [_hiddenLayers removeAllObjects];
    [_allHiddenBiases removeAllObjects];
    [_allHiddenWeights removeAllObjects];
    
    CGFloat _randomMax           = 1.0f;
    CGFloat _randomMin           = -1.0f;
    NSInteger _inputNetCount     = self.countInputNets;
    NSInteger _outputNetCount    = self.countOutputNets;
    NSInteger _layerNetCount     = 1;
    NSInteger _nextLayerNetCount = 1;
    
    //尚未設定任何隱藏層
    if( [_allHiddenBiases count] < 1 && [_allHiddenWeights count] < 1 )
    {
        for( int _layer = 0; _layer < _totalLayers; _layer++ )
        {
            //本層有幾顆
            _layerNetCount     = [self _calculateLayerNetCountWithInputNumber:_inputNetCount outputNumber:_outputNetCount];
            //計算下一層有幾顆
            _inputNetCount     = _layerNetCount;
            _nextLayerNetCount = [self _calculateLayerNetCountWithInputNumber:_inputNetCount outputNumber:_outputNetCount];
            //設定有幾顆隱藏層神經元，就有幾個偏權值
            NSMutableArray *_nextWeights = [NSMutableArray arrayWithCapacity:0];
            for( int i = 0; i < _layerNetCount; i++ )
            {
                //設定有幾條至下一層神經元相映射的權重值
                for( int j = 0; j < _nextLayerNetCount; j++ )
                {
                    [_nextWeights addObject:[NSNumber numberWithFloat:[self _randomMax:_randomMax min:_randomMin]]];
                }
                
                [self addHiddenLayerAtIndex:_layer
                                    netBias:[self _randomMax:_randomMax min:_randomMin]
                                 netWeights:(NSArray *)[_nextWeights copy]];
                
                [_nextWeights removeAllObjects];
            }
            
            //下一層只有 1 顆神經元，之後就不再作事
            if( _nextLayerNetCount <= 1 )
            {
                break;
            }
        }
    }
    
}

-(void)randomInputWeights
{
    //先清空歸零
    [_inputWeights removeAllObjects];
    //單組輸入向量有多長，就有多少顆輸入層神經元
    NSInteger _inputNetCount  = self.countInputNets;
    NSInteger _outputNetCount = self.countOutputNets;
    NSInteger _hiddenNetCount = [self _calculateLayerNetCountWithInputNumber:_inputNetCount outputNumber:_outputNetCount];
    if( [_inputWeights count] < 1 )
    {
        //計算共有幾條隱藏層的權重線
        //NSInteger _totalLines = _hiddenNetCount * _inputNetCount;
        NSMutableArray *_netWeights = [NSMutableArray arrayWithCapacity:0];
        for( int i=0; i<_inputNetCount; i++ )
        {
            for( int j=0; j<_hiddenNetCount; j++ )
            {
                [_netWeights addObject:[NSNumber numberWithDouble:[self _randomMax:1.0f min:-1.0f]]];
            }
            [self addPatternWeights:[_netWeights copy]];
            [_netWeights removeAllObjects];
        }
    }
    
}

-(void)randomWeights
{
    //Input Layer
    [self randomInputWeights];
    
    //Hidden Layers
    [self randomHiddenLayerWeightsWithTotalLayers:[self evaluateHiddenLayerNumbers]];
    
    //輸出層神經元的偏權值
    _outputBias   = [self _randomMax:1.0f min:-1.0f];
    
    //學習速率
    _learningRate = [self _randomMax:1.0f min:0.1f];
}

#pragma --mark Training Public Methods
/*
 * @ Start Training BPN
 *   - Delegate 和 Block 的記憶消耗量在遞迴的實驗下，是一樣的。
 *   - 只單在 dispatch_queue_t 裡跑遞迴，1070 次以後會 Crash，因為 dispatch_queue 的 memory 有限制，改成迭代 1000 次就換下一個 queue 跑 training 就行了。
 */
-(void)training
{
    [self _trainingWithExtraSetupHandler:nil];
}

/*
 * @ Start Training BPN
 *   - And it'll auto save the trained-network when it finished.
 */
-(void)trainingDoneSave
{
    [self _trainingWithExtraSetupHandler:^
    {
        self._isDoneSave = YES;
    }];
}

/*
 * @ Start Training BPN
 *   - It'll random setup all weights and biases.
 */
-(void)trainingWithRandom
{
    [self randomWeights];
    [self training];
}

/*
 * @ Start Training BPN
 *   - It'll random setup all weights and biases, then it'll auto save the trained-network when it finished.
 */
-(void)trainingWithRandomAndSave
{
    self._isDoneSave = YES;
    [self randomWeights];
    [self _trainingWithExtraSetupHandler:^
    {
        self._isDoneSave = YES;
    }];
}

/*
 * @ Pause Training BPN
 *   - It'll force stop, and the trained data will keep in network.
 */
-(void)pause
{
    _isTraining        = NO;
    _forceStopTraining = YES;
}

/*
 * @ Continue training
 */
-(void)continueTraining
{
    _forceStopTraining = NO;
    //[self _formatMaxMultiple];
    [self _startTraining];
}

/*
 * @ Reset to initialization
 */
-(void)reset
{
    [self _resetTrainedParameters];
    [self _recoverOriginalParameters];
}

/*
 * @ 
 *   - 單純使用訓練好的網路作輸出，不跑導傳遞的訓練方法
 */
-(void)useTrainedNetworkToOutput
{
    NSArray *_trainInputs  = [_inputs firstObject];
    dispatch_queue_t queue = dispatch_queue_create("com.krbpn.trained-network", NULL);
    dispatch_async(queue, ^(void){
        [self pause];
        dispatch_async(dispatch_get_main_queue(), ^{
            [self _formatMaxMultiple];
            //將訓練迭代變為 1 次即終止
            //[self recoverTrainedNetwork];
            _limitGeneration    = 1;
            _trainingGeneration = _limitGeneration;
            self._hiddenOutputs = [self _sumHiddenLayerNetWeightsFromInputs:_trainInputs];
            self.outputResults  = [self _sumOutputLayerNetsValue];
            if( self.limitGeneration > 0 && self.trainingGeneration >= self.limitGeneration )
            {
                [self _completedTraining];
                return;
            }
        });
    });
    
}

#pragma --mark Trained Network Public Methods
/*
 * @ Save the trained-network of BPN to NSUserDefaults
 *   - 儲存訓練後的 BPN Network 至 NSUserDefaults
 *   - 同時會保存原訓練的所有 I/O 數據資料
 */
-(void)saveTrainedNetwork
{
    //Does it need to use the [copy] method to copy the whole memory and data into the _bpnNetwork to save in NSUserDefaults ?
    KRBPNTrainedNetwork *_bpnNetwork = [[KRBPNTrainedNetwork alloc] init];
    _bpnNetwork.inputs               = _inputs;
    _bpnNetwork.inputWeights         = _inputWeights;
    _bpnNetwork.hiddenLayers         = _hiddenLayers;
    _bpnNetwork.allHiddenWeights     = _allHiddenWeights;
    _bpnNetwork.allHiddenBiases      = _allHiddenBiases;
    _bpnNetwork.outputBias           = _outputBias;
    _bpnNetwork.outputResults        = _outputResults;
    _bpnNetwork.outputGoals          = _outputGoals;
    _bpnNetwork.learningRate         = _learningRate;
    _bpnNetwork.convergenceError     = _convergenceError;
    _bpnNetwork.fOfAlpha             = _fOfAlpha;
    _bpnNetwork.limitGeneration      = _limitGeneration;
    _bpnNetwork.trainingGeneration   = _trainingGeneration;
    [self removeTrainedNetwork];
    _trainedNetwork                  = _bpnNetwork;
    [NSUserDefaults saveTrainedNetwork:_bpnNetwork forKey:_kTrainedNetworkInfo];
}

/*
 * @ Remove the saved trained-netrowk
 */
-(void)removeTrainedNetwork
{
    [NSUserDefaults removeValueForKey:_kTrainedNetworkInfo];
    _trainedNetwork = nil;
}

/*
 * @ Recovers trained-network data
 *   - 復原訓練過的 BPN Network 數據資料
 */
-(void)recoverTrainedNetwork:(KRBPNTrainedNetwork *)_recoverNetwork
{
    if( _recoverNetwork )
    {
        dispatch_async(dispatch_get_main_queue(), ^
        {
            _inputs             = _recoverNetwork.inputs;
            _inputWeights       = _recoverNetwork.inputWeights;
            _hiddenLayers       = _recoverNetwork.hiddenLayers;
            _allHiddenWeights   = _recoverNetwork.allHiddenWeights;
            _allHiddenBiases    = _recoverNetwork.allHiddenBiases;
            _outputBias         = _recoverNetwork.outputBias;
            _outputResults      = _recoverNetwork.outputResults;
            _outputGoals        = _recoverNetwork.outputGoals;
            _learningRate       = _recoverNetwork.learningRate;
            _convergenceError   = _recoverNetwork.convergenceError;
            _fOfAlpha           = _recoverNetwork.fOfAlpha;
            _limitGeneration    = _recoverNetwork.limitGeneration;
            _trainingGeneration = _recoverNetwork.trainingGeneration;
            [self removeTrainedNetwork];
            _trainedNetwork     = _recoverNetwork;
            [NSUserDefaults saveTrainedNetwork:_trainedNetwork forKey:_kTrainedNetworkInfo];
        });
    }
}

/*
 * @ Recovers saved trained-network of BPN
 *   - 復原已儲存的訓練過的 BPN Network 數據資料
 */
-(void)recoverTrainedNetwork
{
    //[self recoverTrainedNetwork:_trainedNetwork];
    [self recoverTrainedNetwork:self.trainedNetwork];
}

#pragma --mark Blocks
-(void)setTrainingCompletion:(KRBPNTrainingCompletion)_theBlock
{
    _trainingCompletion = _theBlock;
}

-(void)setEachGeneration:(KRBPNEachGeneration)_theBlock
{
    _eachGeneration     = _theBlock;
}

#pragma --mark Getters
-(NSDictionary *)trainedInfo
{
    //代表有不落在 -1.0 ~ 1.0 之間的值，就回復至原先的數值長度與型態
    if( self._maxMultiple != 1 )
    {
        NSMutableArray *_formatedOutputResults = [NSMutableArray new];
        for( NSArray *_groupResults in _outputResults )
        {
            for( NSNumber *_result in _groupResults )
            {
                //還原每一個 goalValue 當初設定的原同等位同寬度的結果值
                double _recoveredRetuls = [_result doubleValue] * self._maxMultiple;
                [_formatedOutputResults addObject:[NSNumber numberWithDouble:_recoveredRetuls]];
            }
        }
        self.outputResults     = nil;
        _outputResults         = [[NSArray alloc] initWithArray:_formatedOutputResults];
        [_formatedOutputResults removeAllObjects];
        _formatedOutputResults = nil;
    }
    
    return @{KRBPNTrainedInfoInputWeights      : _inputWeights,
             KRBPNTrainedInfoHiddenLayers      : _hiddenLayers,
             KRBPNTrainedInfoHiddenWeights     : _allHiddenWeights,
             KRBPNTrainedInfoHiddenBiases      : _allHiddenBiases,
             KRBPNTrainedInfoOutputBias        : [NSNumber numberWithDouble:_outputBias],
             KRBPNTrainedInfoOutputResults     : _outputResults,
             KRBPNTrainedInfoTrainedGeneration : [NSNumber numberWithInteger:_trainingGeneration]};
}

-(KRBPNTrainedNetwork *)trainedNetwork
{
    if( !_trainedNetwork )
    {
        _trainedNetwork = [NSUserDefaults trainedNetworkValueForKey:_kTrainedNetworkInfo];
        if ( !_trainedNetwork )
        {
            return nil;
        }
    }
    return _trainedNetwork;
}

//計算 Hidden Layer 共有幾層
-(NSInteger)countHiddenLayers
{
    return [_hiddenLayers count];
}

//計算 Output Layer 的 Nets 有幾顆
-(NSInteger)countOutputNets
{
    /*
     * !@ 這裡不精簡寫判斷式，是為了將來擴充時的彈性
     */
    NSInteger _count = 0;
    //是多輸出
    if( [[_outputGoals firstObject] isKindOfClass:[NSArray class]] )
    {
        _count = [[_outputGoals firstObject] count];
    }
    else
    {
        //單輸出
        _count = 1;
    }
    return _count;
}

//計算 Input Layer 的 Nets 有幾顆
-(NSInteger)countInputNets
{
    if( !_inputs )
    {
        return 0; //-1;
    }
    return [[_inputs firstObject] count];
}

@end


