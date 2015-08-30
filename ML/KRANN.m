//
//  KRANN.m
//  ANN V2.1.4
//
//  Created by Kalvar on 13/6/28.
//  Copyright (c) 2013 - 2015年 Kuo-Ming Lin (Kalvar Lin, ilovekalvar@gmail.com). All rights reserved.
//
/*
 * @ 常使用的 f(x) 轉換函式為「雙彎曲函數」= 1 / ( 1 + e^-x )
 *   - 須將輸入的值域定在 [0.0, 1.0] 之間
 *   - 輸出則為 {0.0, 1.0} 2 種正負訊號
 */

#import "KRANN.h"
#import "KRANN+NSUserDefaults.h"

//Advises the max nets count per hidden layer
#define ADVISE_MAX_HIDDEN_NETS_COUNT  10
//Advises the max hidden layer count
#define ADVISE_MAX_HIDDEN_LAYER_COUNT 3
#define RANDOM_WEIGHT_MAX             0.5f
#define RANDOM_WEIGHT_MIN            -0.5f

static NSString *_kOriginalInputs           = @"_kOriginalInputs";
static NSString *_kOriginalInputWeights     = @"_kOriginalInputWeights";
static NSString *_kOriginalHiddenLayers     = @"_kOriginalHiddenLayers";
static NSString *_kOriginalAllHiddenWeights = @"_kOriginalAllHiddenWeights";
static NSString *_kOriginalAllHiddenBiases  = @"_kOriginalAllHiddenBiases";
static NSString *_kOriginalOutputBiases     = @"_kOriginalOutputBiases";
static NSString *_kOriginalOutputResults    = @"_kOriginalOutputResults";
static NSString *_kOriginalOutputGoals      = @"_kOriginalOutputGoals";
static NSString *_kOriginalLearningRate     = @"_kOriginalLearningRate";
static NSString *_kOriginalConvergenceError = @"_kOriginalConvergenceError";
static NSString *_kOriginalFOfAlpha         = @"_kOriginalFOfAlpha";
static NSString *_kOriginalLimitIterations = @"_kOriginalLimitIterations";
//static NSString *_kOriginalMaxMultiple    = @"_kOriginalMaxMultiple";

static NSString *_kTrainedNetworkInfo       = @"kTrainedNetworkInfo";

@interface KRANN ()

//隱藏層的輸出值
@property (nonatomic, strong) NSArray *_hiddenOutputs;
//當前資料的輸出期望值
@property (nonatomic, assign) NSArray *_goalValues;
//輸出層的誤差值
@property (nonatomic, strong) NSArray *_outputErrors;
//是否強制中止訓練
@property (nonatomic, assign) BOOL _forceStop;
//原來的設定值
@property (nonatomic, strong) NSMutableDictionary *_originalParameters;
//訓練完就儲存至 NSUserDefaults 裡
@property (nonatomic, assign) BOOL _isDoneSave;
//記錄當前訓練到哪一組 Input 數據
@property (nonatomic, assign) NSInteger _patternIndex;
//在訓練 goalValue 且其值不在 0.0f ~ 1.0f 之間時，就使用本值進行相除與回乘原同類型值的動作
@property (nonatomic, assign) NSInteger _maxMultiple;
//儲存要用於比較計算 _maxMultiple 值的所有 Target Outputs
@property (nonatomic, strong) NSMutableArray *_compareTargets;
//儲存每一次 訓練迭代裡的每一個輸出誤差值, for MSE / RMSE
@property (nonatomic, strong) NSMutableArray *_patternErrors;
//儲存每一個迭代的誤差總和
//@property (nonatomic, strong) NSMutableArray *_iterationErrors;

@end

@implementation KRANN (fixInitials)

-(void)_resetTrainedParameters
{
    self.outputResults       = nil;
    
    self.trainedNetwork      = nil;
    self.trainingIteration  = 0;
    
    self._hiddenOutputs      = nil;
    self._outputErrors       = nil;
    self._forceStop          = NO;
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
    self.hiddenLayerCount    = 1;
    
    self.countHiddenLayers   = 0;
    self.outputBiases        = [NSMutableArray new];
    self.outputGoals         = [NSMutableArray new];
    self.learningRate        = 0.8f;
    self.convergenceError    = 0.001f;
    self.fOfAlpha            = 1;
    self.limitIteration     = 0;
    self.isTraining          = NO;
    self.trainedInfo         = nil;
    
    self.activationFunction  = KRANNActivationFunctionSigmoid;
    //self.openDebug         = false;
    
    self.trainingCompletion  = nil;
    self.perIteration      = nil;
    
    [self _resetTrainedParameters];
    
    self._maxMultiple        = 1;
    self._goalValues         = nil;
    self._originalParameters = [NSMutableDictionary new];
    self._compareTargets     = [NSMutableArray new];
    self._patternErrors      = [NSMutableArray new];
    //self._iterationErrors  = [NSMutableArray new];
    
}

@end

@implementation KRANN (fixMethods)

-(void)_stopTraining
{
    self.isTraining = NO;
}

-(void)_completedTraining
{
    self.isTraining = NO;
    if( self._isDoneSave )
    {
        self._isDoneSave = NO;
        [self saveNetwork];
    }
    
    if( self.delegate )
    {
        if( [self.delegate respondsToSelector:@selector(krANNDidTrainFinished:trainedInfo:totalTimes:)] )
        {
            [self.delegate krANNDidTrainFinished:self trainedInfo:self.trainedInfo totalTimes:self.trainingIteration];
        }
    }
    
    if( self.trainingCompletion )
    {
        self.trainingCompletion(YES, self.trainedInfo, self.trainingIteration);
    }
}

-(void)_printEachIteration
{
    if( self.delegate )
    {
        if( [self.delegate respondsToSelector:@selector(krANNPerIteration:trainedInfo:times:)] )
        {
            [self.delegate krANNPerIteration:self trainedInfo:self.trainedInfo times:self.trainingIteration];
        }
    }
    
    if( self.perIteration )
    {
        self.perIteration(self.trainingIteration, self.trainedInfo);
    }
}

-(void)_copyParameters
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
    [_originals setObject:[self.outputBiases copy] forKey:_kOriginalOutputBiases];
    [_originals setObject:[self.outputGoals copy] forKey:_kOriginalOutputGoals];
    [_originals setObject:[NSNumber numberWithFloat:self.learningRate] forKey:_kOriginalLearningRate];
    [_originals setObject:[NSNumber numberWithDouble:self.convergenceError] forKey:_kOriginalConvergenceError];
    [_originals setObject:[NSNumber numberWithFloat:self.fOfAlpha] forKey:_kOriginalFOfAlpha];
    [_originals setObject:[NSNumber numberWithInteger:self.limitIteration] forKey:_kOriginalLimitIterations];
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
            
            [self.outputBiases removeAllObjects];
            [self.outputBiases addObjectsFromArray:[_originals objectForKey:_kOriginalOutputBiases]];
            
            self.learningRate     = [[_originals objectForKey:_kOriginalLearningRate] floatValue];
            self.convergenceError = [[_originals objectForKey:_kOriginalConvergenceError] doubleValue];
            self.fOfAlpha         = [[_originals objectForKey:_kOriginalFOfAlpha] floatValue];
            self.limitIteration  = [[_originals objectForKey:_kOriginalLimitIterations] integerValue];
            
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
     *   
     * @ Samples
     *   - srand((int)time(NULL));
     *     double _random = ((double)rand() / RAND_MAX) * (_maxValue - _minValue) + _minValue;
     *     RAND_MAX 是 0x7fffffff (2147483647)，而 arc4random() 返回的最大值则是 0x100000000 (4294967296)，
     *     故 * 2.0f 待除，或使用自訂義 ARC4RANDOM_MAX      0x100000000
     */
    return ((double)arc4random() / ( RAND_MAX * 2.0f ) ) * (_maxValue - _minValue) + _minValue;
}

-(NSInteger)_formatHiddenNetCount:(NSInteger)_netCount
{
    NSInteger _hiddenNetCount = _netCount;
    if( _hiddenNetCount < 1 )
    {
        //最少 1 顆
        _hiddenNetCount = 1;
    }
    
    /*
    //隱藏層神經元 MAX is 20 or device will overloading.
    if( _hiddenNetCount > ADVISE_MAX_HIDDEN_NETS_COUNT )
    {
        _hiddenNetCount = ADVISE_MAX_HIDDEN_NETS_COUNT;
    }
    else if( _hiddenNetCount < 1 )
    {
        //最少 1 顆
        _hiddenNetCount = 1;
    }
     */
    
    return _hiddenNetCount;
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
            //是第幾層的隱藏層
            ++_layerNumber;
            //隱藏層每一顆 Net 的偏權值
            NSMutableArray *_hiddenBiases = [NSMutableArray new];
            //輸入向量數據到指定隱藏層間的權重值(線)
            NSMutableArray *_inputWeights = [NSMutableArray new];
            //列舉每一層隱藏層裡的所有 Nets Infomation
            for( NSArray *_eachNets in _eachHiddenLayers )
            {
                //取出 Hidden Net 的偏權值 (型態是 NSNumber)
                [_hiddenBiases addObject:[_eachNets firstObject]];
                
                //取出 Hidden Net 的 Input to next Net 的權重值 (型態是 NSArray)
                [_inputWeights addObject:[_eachNets lastObject]];
            }
            
            /*
             * @ 以下都會變成 3 層陣列
                 allHiddenBiases = @[
                    //Hidden Layer 1
                    @[
                        //Net 4
                        @0.3,
                        //Net 5
                        @0.2
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

@implementation KRANN (fixFOfNets)
/*
 * @ S 形函數
 *   - [0.0, 1.0]
 */
-(float)_fOfSigmoid:(float)_x
{
    return ( 1.0f / ( 1 + powf(M_E, (-(self.fOfAlpha) * _x)) ) );
}

/*
 * @ 雙曲線函數
 *   - [-1.0, 1.0]
 */
-(float)_fOfTanh:(float)_x
{
    return ( 2.0f / ( 1 + powf(M_E, (-2.0f * _x)) ) ) - 1.0f;
}

/*
 * @ Fuzzy function
 *   - Still waiting for implementing.
 */
-(float)_fOfFuzzy:(float)_x
{
    //Do Fuzzy ...
    return -0.1f;
}

-(float)_fOfX:(float)_x
{
    float _y = 0.0f;
    switch (self.activationFunction)
    {
        case KRANNActivationFunctionTanh:
            _y = [self _fOfTanh:_x];
            break;
        case KRANNActivationFunctionFuzzy:
            _y = [self _fOfFuzzy:_x];
            break;
        case KRANNActivationFunctionSigmoid:
            _y = [self _fOfSigmoid:_x];
        default:
            break;
    }
    //isNaN ( not a number )
    /*
    if( _y != _y )
    {
        [self restart];
    }
     */
    return _y;
}

@end

@implementation KRANN (fixTrainings)

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
    //net(j)
    NSMutableArray *_fOfNets = [NSMutableArray new];
    //輸入層要做 SUM 就必須取出同一維度的值做運算，以 Hidden Layer 的神經元為維度來源
    //取出有多少維度
    int _totalDimesion       = [_biases count];
    //直接用維度取值作 SUM
    for( int _dimesion = 0; _dimesion < _totalDimesion; _dimesion++ )
    {
        NSNumber *_netBias   = [_biases objectAtIndex:_dimesion];
        //再以同維度做 SUM 方法
        float _sumOfNet      = 0;
        //有幾個 Input 就有幾個 Weight
        //取出每一個輸入值( Ex : X1 轉置矩陣後的輸入向量 [1, 2, -1] )
        int _inputIndex      = -1;
        for( NSNumber *_xi in _layerInputs )
        {
            ++_inputIndex;
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
        //減同維度的隱藏層神經元偏權值
        _sumOfNet    -= [_netBias floatValue];
        float _fOfNet = [self _fOfX:_sumOfNet];
        [_fOfNets addObject:[NSNumber numberWithFloat:_fOfNet]];
    }
    
    //@[_fOfNet(4), _fOfNet(5), ...]
    return ( [_fOfNets count] > 0 ) ? (NSArray *)_fOfNets : nil;
}

/*
 * @ 所有隱藏層的輸出都在這裡完成，在這裡不斷遞迴運算每一個隱藏層
 */
-(NSArray *)_sumHiddenLayerOutputsFromInputs:(NSArray *)_inputs
{
    //運算完成的 Nets
    NSMutableArray *_toOutputNets = [NSMutableArray new];
    NSArray *_hiddenLayers        = (NSArray *)self.hiddenLayers;
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
                //2 層陣列，依照 Hidden Layer # 存放每一層 Hidden Layer 裡每一顆 Net 作完 SUM 後的 Output 值
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
    NSMutableArray *_fOfNets  = nil;
    //The last hidden layer outputs
    NSArray *_lastHideOutputs = (NSArray *)[self._hiddenOutputs lastObject];
    if( _lastHideOutputs )
    {
        //The last hidden layer outputs
        NSArray *_lastHiddenWeights = [self.allHiddenWeights lastObject];
        _fOfNets                    = [NSMutableArray new];
        int _outputIndex            = -1;
        for( NSNumber *_outputBias in self.outputBiases )
        {
            float _sumOfNet = 0;
            ++_outputIndex;
            int _netIndex   = -1;
            for( NSArray *_weights in _lastHiddenWeights )
            {
                ++_netIndex;
                NSNumber *_outputWeight  = [_weights objectAtIndex:_outputIndex];
                _sumOfNet               += [[_lastHideOutputs objectAtIndex:_netIndex] floatValue] * [_outputWeight floatValue];
            }
            _sumOfNet        -= [_outputBias floatValue];
            float _netOutput  = [self _fOfX:_sumOfNet];
            [_fOfNets addObject:[NSNumber numberWithFloat:_netOutput]];
        }
    }
    return _fOfNets;
}

/*
 * @ 計算輸出層神經元( Net )的輸出誤差 ( 與期望值的誤差 )
 *   - 公式 : Oj x ( 1 - Oj ) x ( Tj - Oj )
 */
-(NSArray *)_calculateOutputError
{
    NSMutableArray *_errors = nil;
    self.outputResults      = [self _sumOutputLayerNetsValue];
    NSArray *_netOutputs    = self.outputResults;
    if( _netOutputs )
    {
        _errors = [NSMutableArray new];
        NSInteger _outputIndex = -1;
        for( NSNumber *_netOutput in _netOutputs )
        {
            ++_outputIndex;
            //取出輸出層神經元的輸出結果
            float _outputValue = [_netOutput floatValue];
            float _targetValue = [[self._goalValues objectAtIndex:_outputIndex] floatValue] / self._maxMultiple;
            //計算與期望值的誤差
            float _errorValue  = _outputValue * ( 1 - _outputValue ) * ( _targetValue - _outputValue );
            [_errors addObject:[NSNumber numberWithFloat:_errorValue]];
            
            //儲存每一個 Pattern 的輸出誤差 (Tj - Yj) ^ 2
            float _mseError    = _targetValue - _outputValue;
            [self._patternErrors addObject:[NSNumber numberWithFloat:(_mseError * _mseError)]];
        }
    }
    return _errors;
}

/*
 * @ 計算所有隱藏層神經元( Nets )的誤差
 *   - 單輸出在用的公式 : Oj x ( 1 - Oj ) x Errork x Wjk
 *
 *   Input Layer   Hidden Layer 1   Hidden Layer 2   Hidden Layer 3   Output Layer
 *
 *     Net 1
 *     Net 2         Net 5            Net 7             Net 9           Net 11
 *     Net 3         Net 6            Net 8             Net 10          Net 12
 *     Net 4                                                            Net 13
 *
 */
-(NSArray *)_calculateNetsError
{
    //取得輸出層輸出誤差 ( 與期望值的誤差 )
    self._outputErrors         = [self _calculateOutputError];
    NSArray *_errors           = self._outputErrors;
    NSMutableArray *_allErrors = nil;
    if( _errors )
    {
        _allErrors                 = [NSMutableArray new];
        //Hidden Layer 1, Hidden Layer 2, ...
        //Ex : @[ @[0.1, 0.2], @[-0.2, -0.3, 0.5], ... ]
        NSArray *_allHiddenOutputs = self._hiddenOutputs;
        NSArray *_allHiddenWeights = self.allHiddenWeights;
        int _hiddenCount           = [_allHiddenOutputs count];
        //倒取每一層隱藏層的 Output 結果
        for( int _layerIndex=_hiddenCount-1; _layerIndex>=0; _layerIndex-- )
        {
            NSArray *_lastErrors   = nil;
            //代表已運算過 Output Layer 的輸出誤差
            if( [_allErrors count] > 0 )
            {
                //改用上一層的 Hidden Layer Output Error 來進行下一層的 Error 運算
                _lastErrors        = [_allErrors lastObject];
            }
            else
            {
                //Use the output layer errors
                _lastErrors        = _errors;
            }
            
            //Current Hidden Layer errors
            NSMutableArray *_netErrors  = [NSMutableArray new];
            //Current Hidden Layer outputs and weights
            NSArray *_layerOutputs      = [_allHiddenOutputs objectAtIndex:_layerIndex];
            NSArray *_hiddenWeights     = [_allHiddenWeights objectAtIndex:_layerIndex];
            int _netIndex               = -1;
            for( NSNumber *_output in _layerOutputs )
            {
                ++_netIndex;
                float _hiddenOutput     = [_output floatValue];
                int _weightIndex        = -1;
                float _sumError         = 0.0f;
                //SUM output layer errors
                for( NSNumber *_outputError in _lastErrors )
                {
                    ++_weightIndex;
                    float _netWeight    = [[[_hiddenWeights objectAtIndex:_netIndex] objectAtIndex:_weightIndex] floatValue];
                    float _hiddenError  = [_outputError floatValue] * _netWeight;
                    _sumError          += _hiddenError;
                }
                //微分, S * Hidden layer net output * ( 1 - Hidden layer net output )
                _sumError *= _hiddenOutput * ( 1 - _hiddenOutput );
                [_netErrors addObject:[NSNumber numberWithFloat:_sumError]];
            }
            [_allErrors addObject:_netErrors];
        }
    }
    //@[ Hidden Layer 3 errors, Hidden Layer 2 errors ... ];
    //@[ @[ @[Net 7 error, Net 8 error], @[Net 4 error, Net 5 error, Net 6 error], ... ] ]
    return _allErrors;
}

/*
 * @ 更新權重與偏權值
 *   - 公式 : Shita(i) = Shita(i) + learning rate x Error(k)
 *              偏權值 = 偏權值    +   學習速率      x 要修改的誤差值
 */
-(BOOL)_refreshNetsWeights
{
    if( self._forceStop )
    {
        [self _stopTraining];
        return NO;
    }
    
    self.isTraining         = YES;
    BOOL _onePatternTrained = NO;
    //隱藏層神經元的輸出誤差值
    //@[ @[Net 9 error], @[Net 7 error, Net 8 error], @[Net 4 error, Net 5 error, Net 6 error], ... ]
    NSArray *_hiddenErrors  = [self _calculateNetsError];
    if( _hiddenErrors )
    {
        //取出共有幾層 Hidden Layer
        NSArray *_allHiddenWeights = self.allHiddenWeights;
        int _hiddenLayerIndex      = [_allHiddenWeights count];
        if( _hiddenLayerIndex < 1 )
        {
            return NO;
        }
        
        //儲存所有 Updated 的 Hidden Layer Weights
        //NSMutableArray *_updatedAllHiddenWeights = [NSMutableArray new];
        
        //更新輪出層到隱藏層的偏權值, update biases of output layer.
        NSMutableArray *_updatedOutputBiases     = [NSMutableArray new];
        int _outputIndex                         = -1;
        for( NSNumber *_outputError in self._outputErrors )
        {
            ++_outputIndex;
            float _targetError  = [_outputError floatValue];
            float _outputBias   = [[self.outputBiases objectAtIndex:_outputIndex] floatValue];
            _outputBias        += ( self.learningRate * _targetError );
            [_updatedOutputBiases addObject:[NSNumber numberWithFloat:_outputBias]];
        }
        [self.outputBiases removeAllObjects];
        [self.outputBiases addObjectsFromArray:_updatedOutputBiases];
        _updatedOutputBiases    = nil;
        
        //更新隱藏層到輸出層的權重
        --_hiddenLayerIndex;
        NSArray *_lastHiddenWeights               = [_allHiddenWeights objectAtIndex:_hiddenLayerIndex];
        NSArray *_lastHiddenOutputs               = [self._hiddenOutputs objectAtIndex:_hiddenLayerIndex];
        NSMutableArray *_updatedLastHiddenWeights = [NSMutableArray new];
        int _netIndex                             = -1;
        for( NSArray *_netWeights in _lastHiddenWeights )
        {
            ++_netIndex;
            float _hiddenOutput            = [[_lastHiddenOutputs objectAtIndex:_netIndex] floatValue];
            NSMutableArray *_newNetWeights = [NSMutableArray new];
            //對應到哪一個 Output Layer Net 的 Error Value
            int _errorIndex                = -1;
            for( NSNumber *_netWeight in _netWeights )
            {
                ++_errorIndex;
                float _targetError  = [[self._outputErrors objectAtIndex:_errorIndex] floatValue];
                float _weight       = [_netWeight floatValue] + ( self.learningRate * _targetError * _hiddenOutput );
                //原公式在精度上較差，故暫不採用 : learning rate * last error value * last output value
                //float _weight     = self.learningRate * _targetError * _hiddenOutput;
                [_newNetWeights addObject:[NSNumber numberWithFloat:_weight]];
            }
            //放入 Last Hidden Layer 裡
            [_updatedLastHiddenWeights addObject:_newNetWeights];
        }
        //[_updatedAllHiddenWeights addObject:_updatedLastHiddenWeights];
        [self.allHiddenWeights replaceObjectAtIndex:_hiddenLayerIndex withObject:_updatedLastHiddenWeights];
        
        //修正隱藏層對隱藏層的權重
        //有幾層 hidden Layers 就跑幾次 ( 從最後一層的 Hidden Errors 取到第一層的 Hidden Errors )
        int _currentIndex = [_hiddenErrors count];
        int _prevIndex    = _currentIndex - 1;
        for( NSArray *_currentErrors in _hiddenErrors )
        {
            --_currentIndex;
            //更新 Hidden Layer 的 Biases
            NSMutableArray *_updatedHiddenBiases = [NSMutableArray new];
            NSArray *_currentHiddenBiases        = [[self.allHiddenBiases objectAtIndex:_currentIndex] copy];
            int _hiddenIndex                     = -1;
            for( NSNumber *_outputError in _currentErrors )
            {
                ++_hiddenIndex;
                float _targetError  = [_outputError floatValue];
                float _hiddenBias   = [[_currentHiddenBiases objectAtIndex:_hiddenIndex] floatValue];
                //修正 Hidden Layer Net Bias
                //原使用公式 : 精度稍高，收斂慢
                //float _updatedBias = _hiddenBias + ( self.learningRate * _sumNetError );
                _hiddenBias        += ( self.learningRate * _targetError );
                [_updatedHiddenBiases addObject:[NSNumber numberWithFloat:_hiddenBias]];
            }
            [self.allHiddenBiases replaceObjectAtIndex:_currentIndex withObject:_updatedHiddenBiases];
            _currentHiddenBiases = nil;
            
            --_prevIndex;
            //更新 Hidden Layer 的 Weights
            if( _prevIndex >= 0 )
            {
                NSArray *_lastHiddenWeights               = [_allHiddenWeights objectAtIndex:_prevIndex];
                NSArray *_lastHiddenOutputs               = [self._hiddenOutputs objectAtIndex:_prevIndex];
                NSMutableArray *_updatedLastHiddenWeights = [NSMutableArray new];
                int _netIndex                             = -1;
                for( NSArray *_netWeights in _lastHiddenWeights )
                {
                    ++_netIndex;
                    float _hiddenOutput            = [[_lastHiddenOutputs objectAtIndex:_netIndex] floatValue];
                    NSMutableArray *_newNetWeights = [NSMutableArray new];
                    //對應到哪一個 Current Hidden Layer Net 的 Error Value
                    int _errorIndex                = -1;
                    for( NSNumber *_netWeight in _netWeights )
                    {
                        ++_errorIndex;
                        float _targetError  = [[_currentErrors objectAtIndex:_errorIndex] floatValue];
                        float _weight       = [_netWeight floatValue] + ( self.learningRate * _targetError * _hiddenOutput );
                        //原公式在精度上較差，故暫不採用 : learning rate * last error value * last output value
                        //float _weight     = self.learningRate * _targetError * _hiddenOutput;
                        [_newNetWeights addObject:[NSNumber numberWithFloat:_weight]];
                    }
                    //放入 Last Hidden Layer 裡
                    [_updatedLastHiddenWeights addObject:_newNetWeights];
                }
                [self.allHiddenWeights replaceObjectAtIndex:_prevIndex withObject:_updatedLastHiddenWeights];
            }
            else
            {
                //_prevIndex < 0
                //最後修正 Input Layer to First Hidden Layer 之間的權重
                NSMutableArray *_updatedInputWeights = [NSMutableArray new];
                NSArray *_weights = [self.inputWeights copy];
                int _inputIndex   = -1;
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
                        float _hiddenError = [[_currentErrors objectAtIndex:_weightIndex] floatValue];
                        float _inputValue  = [[[self.inputs objectAtIndex:self._patternIndex] objectAtIndex:_inputIndex] floatValue];
                        float _resetWeight = _netWeight + ( self.learningRate * _hiddenError * _inputValue );
                        [_resetWeights addObject:[NSNumber numberWithFloat:_resetWeight]];
                        //NSLog(@"_new weight : %f = %f + ( %f * %f * %f )", _resetWeight, _netWeight, self.learningRate, _hiddenError, _inputValue);
                    }
                    //修正 InputWeights 輸入層到第一層隱藏層的權重
                    //[self.inputWeights replaceObjectAtIndex:_inputIndex withObject:_resetWeights];
                    [_updatedInputWeights addObject:_resetWeights];
                }
                [self.inputWeights removeAllObjects];
                [self.inputWeights addObjectsFromArray:_updatedInputWeights]; //[_updatedInputWeights copy];
                [_updatedInputWeights removeAllObjects];
                _updatedInputWeights = nil;
                _weights             = nil;
            }
        }
        _onePatternTrained = YES;
    }
    return _onePatternTrained;
}

-(void)_formatMaxMultiple
{
    //先找出期望值的最大絕對值
    NSNumber *_max  = [self._compareTargets valueForKeyPath:@"@max.self"];
    NSNumber *_min  = [self._compareTargets valueForKeyPath:@"@min.self"];
    double _fabsMax = fabs(_max.doubleValue);
    double _fabsMin = fabs(_min.doubleValue);
    double _realMax = MAX(_fabsMax, _fabsMin);
    if( _realMax > 1.0f )
    {
        self._maxMultiple  = 10 * ( (int)log10(_realMax) + 1 );
    }
}

//均方誤差
-(BOOL)_mse
{
    BOOL _isDone = NO;
    if( [self._patternErrors count] > 0 )
    {
        float _sumOutputError = 0.0f;
        for( NSNumber *_outpurError in self._patternErrors )
        {
            //使用絕對值來做誤差比較
            _sumOutputError += [_outpurError floatValue];
        }
        
        //MSE
        _sumOutputError = _sumOutputError / ([self.inputs count] * [self.outputBiases count]);
        //NSLog(@"mse sumOutputError : %f", _sumOutputError);
        //NSLog(@"\n\n\n");
        //如果已達收斂誤差，就不再繼續訓練
        _isDone = ( _sumOutputError <= self.convergenceError );
    }
    return _isDone;
}

//均方根誤差
-(BOOL)_rmse
{
    BOOL _isDone = NO;
    if( [self._patternErrors count] > 0 )
    {
        float _sumOutputError = 0.0f;
        for( NSNumber *_outpurError in self._patternErrors )
        {
            //使用絕對值來做誤差比較
            _sumOutputError += [_outpurError floatValue];
        }
        
        //RMSE
        _sumOutputError = sqrtf(_sumOutputError / ([self.inputs count] * [self.outputBiases count]));
        //NSLog(@"rmse sumOutputError : %f", _sumOutputError);
        //NSLog(@"\n\n\n");
        _isDone = ( _sumOutputError <= self.convergenceError );
    }
    return _isDone;
}

-(void)_startTraining
{
    ++self.trainingIteration;
    self._patternIndex = -1;
    //開始代入 X1, X2 ... Xn 各組的訓練資料
    for( NSArray *_inputs in self.inputs )
    {
        ++self._patternIndex;
        /*
         * @ 不論正負號都先轉成絕對值，我只要求得除幾位數變成小數點
         *   - NSLog(@"%i", (int)log10f(-81355.555)); //-2147483648
         *   - NSLog(@"%i", (int)log10f(81355.555));  //4 個 10 倍
         *
         */
        self._goalValues    = [self.outputGoals objectAtIndex:self._patternIndex];
        self._hiddenOutputs = [self _sumHiddenLayerOutputsFromInputs:_inputs];
        //更新權重失敗，代表訓練異常，中止 !
        if ( ![self _refreshNetsWeights] )
        {
            //考慮，是否要記錄訓練到哪一筆，等等再繼續 ?
            //要繼續的話應該要重頭再來才是 ?
            break;
        }
    }
    
    //如有指定迭代數 && 當前訓練迭代數 >= 指定迭代數
    if( self.limitIteration > 0 && self.trainingIteration >= self.limitIteration )
    {
        //停止訓練
        [self _completedTraining];
        return;
    }
    
    BOOL _isDone = [self _mse];
    [self._patternErrors removeAllObjects];
    //是否收斂
    if( _isDone )
    {
        //達到收斂誤差值或出現異常狀況，即停止訓練
        [self _completedTraining];
        return;
    }
    else
    {
        //全部數據都訓練完了，才為 1 迭代
        [self _printEachIteration];
        //未達收斂誤差，則繼續執行訓練
        [self _startTraining];
    }
}

-(void)_trainingWithExtraHandler:(void(^)())_extraHandler
{
    //DISPATCH_QUEUE_CONCURRENT
    dispatch_queue_t queue = dispatch_queue_create("com.KRANN.train-network", NULL);
    dispatch_async(queue, ^(void)
    {
        [self pause];
        [self _resetTrainedParameters];
        [self _initHiddenLayerBiasesAndWeights];
        if( _extraHandler )
        {
            _extraHandler();
        }
        [self _copyParameters];
        dispatch_async(dispatch_get_main_queue(), ^
        {
            [self _formatMaxMultiple];
            [self _startTraining];
        });
    });
}

@end

@implementation KRANN

@synthesize delegate            = _delegate;

@synthesize inputs              = _inputs;
@synthesize inputWeights        = _inputWeights;
@synthesize allHiddenWeights    = _allHiddenWeights;
@synthesize allHiddenBiases     = _allHiddenBiases;

@synthesize hiddenLayers        = _hiddenLayers;
@synthesize hiddenLayerCount    = _hiddenLayerCount;

@synthesize countHiddenLayers   = _countHiddenLayers;
@synthesize countOutputNets     = _countOutputNets;
@synthesize countInputNets      = _countInputNets;

@synthesize outputBiases        = _outputBiases;
@synthesize outputResults       = _outputResults;
@synthesize outputGoals         = _outputGoals;
@synthesize learningRate        = _learningRate;
@synthesize convergenceError    = _convergenceError;
@synthesize fOfAlpha            = _fOfAlpha;
@synthesize limitIteration     = _limitIteration;
@synthesize trainingIteration  = _trainingIteration;
@synthesize isTraining          = _isTraining;
@synthesize trainedInfo         = _trainedInfo;
@synthesize trainedNetwork      = _trainedNetwork;

@synthesize activationFunction  = _activationFunction;
//@synthesize openDebug         = _openDebug;

@synthesize trainingCompletion  = _trainingCompletion;
@synthesize perIteration      = _perIteration;

@synthesize _hiddenOutputs;
@synthesize _goalValues;
@synthesize _outputErrors;
@synthesize _forceStop;
@synthesize _originalParameters;
@synthesize _isDoneSave;
@synthesize _patternIndex;
@synthesize _maxMultiple;
@synthesize _compareTargets;
@synthesize _patternErrors;

+(instancetype)sharedNetwork
{
    static dispatch_once_t pred;
    static KRANN *_object = nil;
    dispatch_once(&pred, ^
    {
        _object = [[KRANN alloc] init];
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
 * @params, _patterns : 各輸入向量陣列值
 * @params, _weights  : 輸入層各向量值到隱藏層神經元的權重
 * @params, _goals    : 每一筆輸入向量的期望值( 輸出期望 )
 */
-(void)addPatterns:(NSArray *)_patterns outputGoals:(NSArray *)_goals
{
    [_inputs addObject:_patterns];
    [_outputGoals addObject:_goals]; //@[ @[1.0, 0.8, 0.2], @[0.76, 0.5, 0.89], ... ]
    [self._compareTargets addObjectsFromArray:_goals];
}

/*
 * @ 輸入層各向量值到隱藏層神經元的權重
 *   - 連結同一個 Net 的就一組一組分開，有幾個 Hidden Net 就會有幾組
 */
-(void)addPatternWeights:(NSArray *)_weights
{
    [_inputWeights addObject:_weights]; //@[ @[W14, W15], @[W24, W25], @[W34, W35] ]
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
         @[
           //Net 4, [隱藏層神經元 Net 4 的偏權值, @[隱藏層神經元 Net 4 到下一層神經元的權重值]]
           @[@-0.4, @[@-0.3, @0.2]],
           //Net 5
           @[@0.2, @[@-0.2, @0.5]]
         ],
         
         //Hidden Layer 2
         @[
           //Net 6
           @[@0.3, @[@-0.5, @0.1]],
           //Net 7
           @[@0.7, @[@0.2, @0.4]]
         ],
     
         nil];
    */
    
    NSMutableArray *_layers = nil;
    NSMutableArray *_nets   = [NSMutableArray new];
    [_nets addObject:[NSNumber numberWithFloat:_netBias]];
    [_nets addObject:_netWeights];
    
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

/*
 * @ 設定 Output Layer Nets 的 Biases
 */
-(void)addOutputBiases:(NSArray *)_biases
{
    [_outputBiases addObjectsFromArray:_biases];
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
    NSInteger _layerNumber = 0;
    //如果有設定想要幾個隱藏層
    if( _hiddenLayerCount > 0 )
    {
        _layerNumber = _hiddenLayerCount;
    }
    else
    {
        //沒有設定想要幾個隱藏層，就讓系統來決定層數
        NSInteger _inputNetCount  = self.countInputNets;
        NSInteger _outputNetCount = self.countOutputNets;
        NSInteger _layerNetCount  = [self _calculateLayerNetCountWithInputNumber:_inputNetCount outputNumber:_outputNetCount];
        for( int i = 0; i<_layerNetCount; i++ )
        {
            //本層有幾顆
            _layerNetCount   = [self _calculateLayerNetCountWithInputNumber:_inputNetCount outputNumber:_outputNetCount];
            if( _layerNetCount <= _outputNetCount )
            {
                break;
            }
            
            ++_layerNumber;
            
            if( _layerNumber > ADVISE_MAX_HIDDEN_LAYER_COUNT )
            {
                _layerNumber = ADVISE_MAX_HIDDEN_LAYER_COUNT;
                break;
            }
            
            //計算下一層有幾顆
            _inputNetCount   = _layerNetCount;
        }

    }
    return _layerNumber;
}

/*
 * @ Random all hidden-net weights, net biases, output net bias.
 *   - 亂數設定隱藏層神經元的權重、神經元偏權值、輸出層神經元偏權值
 */
-(void)randomHiddenWeightsWithTotalLayers:(int)_totalLayers
{
    //At least 1 hidden layer.
    if( _totalLayers < 1 )
    {
        _totalLayers = 1;
    }
    
    //先清空舊資料
    [_hiddenLayers removeAllObjects];
    [_allHiddenBiases removeAllObjects];
    [_allHiddenWeights removeAllObjects];
    
    CGFloat _randomMax           = RANDOM_WEIGHT_MAX;
    CGFloat _randomMin           = RANDOM_WEIGHT_MIN;
    NSInteger _inputNetCount     = self.countInputNets;
    NSInteger _outputNetCount    = self.countOutputNets;
    NSInteger _layerNetCount     = 1;
    NSInteger _nextLayerNetCount = 1;
    
    //NSLog(@"_inputNetCount : %i", _inputNetCount);
    //NSLog(@"_outputNetCount : %i", _outputNetCount);
    
    //尚未設定任何隱藏層
    if( [_allHiddenBiases count] < 1 && [_allHiddenWeights count] < 1 )
    {
        for( int _layer = 0; _layer < _totalLayers; _layer++ )
        {
            //本層有幾顆
            _layerNetCount     = [self _calculateLayerNetCountWithInputNumber:_inputNetCount outputNumber:_outputNetCount];
            //NSLog(@"_layerNetCount : %i", _layerNetCount);
            
            //計算下一層有幾顆
            //是最後一層
            if( _layer == _totalLayers - 1 )
            {
                //指定為對廧 Output Layer Nets 數量
                _nextLayerNetCount = _outputNetCount;
            }
            else
            {
                //減少下一次的神經元數量，以避免過過訓練和落入局部解
                _inputNetCount      = powf(_layerNetCount, 0.5f);
                _nextLayerNetCount  = [self _calculateLayerNetCountWithInputNumber:_inputNetCount outputNumber:_outputNetCount];
            }
            
            //NSLog(@"_nextLayerNetCount : %i", _nextLayerNetCount);
            
            //設定有幾顆隱藏層神經元，就有幾個偏權值
            float _hiddenMax   = _randomMax / _layerNetCount;
            float _hiddenMin   = _randomMin / _layerNetCount;
            NSMutableArray *_nextWeights = [NSMutableArray arrayWithCapacity:0];
            for( int i = 0; i < _layerNetCount; i++ )
            {
                //設定有幾條至下一層神經元相映射的權重值
                for( int j = 0; j < _nextLayerNetCount; j++ )
                {
                    [_nextWeights addObject:[NSNumber numberWithFloat:[self _randomMax:_hiddenMax min:_hiddenMin]]];
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
    
    //NSLog(@"all hidden layers count : %i", [self.hiddenLayers count]);
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
        //權重初始化規則 : ( 0.5 / 此層神經元個數 ) ~ ( -0.5 / 此層神經元個數 )，其它層以此類推
        float _inputMax = RANDOM_WEIGHT_MAX / _inputNetCount;
        float _inputMin = RANDOM_WEIGHT_MIN / _inputNetCount;
        NSMutableArray *_netWeights = [NSMutableArray arrayWithCapacity:0];
        for( int i=0; i<_inputNetCount; i++ )
        {
            for( int j=0; j<_hiddenNetCount; j++ )
            {
                [_netWeights addObject:[NSNumber numberWithDouble:[self _randomMax:_inputMax min:_inputMin]]];
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
    [self randomHiddenWeightsWithTotalLayers:[self evaluateHiddenLayerNumbers]];
    
    //輸出層神經元的偏權值
    NSInteger _outputCount    = [_outputBiases count];
    NSInteger _outputNetCount = [[_outputGoals firstObject] count];
    if( _outputNetCount < 1 )
    {
        _outputNetCount = 1;
    }
    if( _outputCount < _outputNetCount )
    {
        //輸出層神經元的偏權值
        for( int _i=0; _i<_outputNetCount; _i++ )
        {
            [_outputBiases addObject:[NSNumber numberWithDouble:[self _randomMax:RANDOM_WEIGHT_MAX min:RANDOM_WEIGHT_MIN]]];
        }
    }
    
    //學習速率
    //_learningRate = [self _randomMax:1.0f min:0.1f];
}

#pragma --mark Training Public Methods
/*
 * @ Start Training ANN
 *   - Delegate 和 Block 的記憶消耗量在遞迴的實驗下，是一樣的。
 *   - 只單在 dispatch_queue_t 裡跑遞迴，1070 次以後會 Crash，因為 dispatch_queue 的 memory 有限制，改成迭代 1000 次就換下一個 queue 跑 training 就行了。
 */
-(void)training
{
    [self _trainingWithExtraHandler:nil];
}

/*
 * @ Start Training ANN
 *   - And it'll auto save the trained-network when it finished.
 */
-(void)trainingBySave
{
    [self _trainingWithExtraHandler:^
    {
        self._isDoneSave = YES;
    }];
}

/*
 * @ Start Training ANN
 *   - It'll random setup all weights and biases.
 */
-(void)trainingByRandomSettings
{
    [self randomWeights];
    [self training];
}

/*
 * @ Start Training ANN
 *   - It'll random setup all weights and biases, then it'll auto save the trained-network when it finished.
 */
-(void)trainingByRandomWithSave
{
    self._isDoneSave = YES;
    [self randomWeights];
    [self _trainingWithExtraHandler:^
    {
        self._isDoneSave = YES;
    }];
}

/*
 * @ Continue training besides with adding pattern
 */
-(void)trainingWithAddPatterns:(NSArray *)_patterns outputGoals:(NSArray *)_goals
{
    [self addPatterns:_patterns outputGoals:_goals];
    [self continueTraining];
}

/*
 * @ Pause Training ANN
 *   - It'll force stop, and the trained data will keep in network.
 */
-(void)pause
{
    _isTraining        = NO;
    _forceStop = YES;
}

/*
 * @ Continue training
 */
-(void)continueTraining
{
    _forceStop = NO;
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

-(void)restart
{
    [self pause];
    [self reset];
    [self training];
}

/*
 * @ 單純使用訓練好的網路作輸出，不跑導傳遞修正網路
 */
-(void)directOutputAtInputs:(NSArray *)_rawInputs completion:(void(^)())_completion
{
    if( _rawInputs != nil )
    {
        [_inputs removeAllObjects];
        [_inputs addObject:_rawInputs];
    }
    //取出 Inputs 的第 1 筆 Raw Data
    NSArray *_trainInputs  = [_inputs firstObject];
    dispatch_queue_t queue = dispatch_queue_create("com.KRANN.trained-network", NULL);
    dispatch_async(queue, ^(void){
        [self pause];
        dispatch_async(dispatch_get_main_queue(), ^{
            [self _formatMaxMultiple];
            //將訓練迭代變為 1 次即終止
            //[self recoverTrainedNetwork];
            _limitIteration    = 1;
            _trainingIteration = _limitIteration;
            self._hiddenOutputs = [self _sumHiddenLayerOutputsFromInputs:_trainInputs];
            self.outputResults  = [self _sumOutputLayerNetsValue];
            if( self.limitIteration > 0 && self.trainingIteration >= self.limitIteration )
            {
                [self _completedTraining];
                if( _completion )
                {
                    _completion();
                }
                return;
            }
        });
    });
}

-(void)directOutputAtInputs:(NSArray *)_rawInputs
{
    [self directOutputAtInputs:_rawInputs completion:nil];
}

#pragma --mark Trained Network Public Methods
/*
 * @ Save the trained-network of ANN to NSUserDefaults
 *   - 儲存訓練後的 ANN Network 至 NSUserDefaults
 *   - 同時會保存原訓練的所有 I/O 數據資料
 */
-(void)saveNetwork
{
    //Does it need to use the [copy] method to copy the whole memory and data into the _ANNNetwork to save in NSUserDefaults ?
    KRANNTrainedNetwork *_ANNNetwork = [[KRANNTrainedNetwork alloc] init];
    _ANNNetwork.inputs               = _inputs;
    _ANNNetwork.inputWeights         = _inputWeights;
    _ANNNetwork.hiddenLayers         = _hiddenLayers;
    _ANNNetwork.allHiddenWeights     = _allHiddenWeights;
    _ANNNetwork.allHiddenBiases      = _allHiddenBiases;
    _ANNNetwork.outputBiases         = _outputBiases;
    _ANNNetwork.outputResults        = _outputResults;
    _ANNNetwork.outputGoals          = _outputGoals;
    _ANNNetwork.learningRate         = _learningRate;
    _ANNNetwork.convergenceError     = _convergenceError;
    _ANNNetwork.fOfAlpha             = _fOfAlpha;
    _ANNNetwork.limitIteration      = _limitIteration;
    _ANNNetwork.trainingIteration   = _trainingIteration;
    [self removeNetwork];
    _trainedNetwork                  = _ANNNetwork;
    [NSUserDefaults saveTrainedNetwork:_ANNNetwork forKey:_kTrainedNetworkInfo];
}

/*
 * @ Remove the saved trained-netrowk
 */
-(void)removeNetwork
{
    [NSUserDefaults removeValueForKey:_kTrainedNetworkInfo];
    _trainedNetwork = nil;
}

/*
 * @ Recovers trained-network data
 *   - 復原訓練過的 ANN Network 數據資料
 */
-(void)recoverNetwork:(KRANNTrainedNetwork *)_recoverNetwork
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
            _outputBiases       = _recoverNetwork.outputBiases;
            _outputResults      = _recoverNetwork.outputResults;
            _outputGoals        = _recoverNetwork.outputGoals;
            _learningRate       = _recoverNetwork.learningRate;
            _convergenceError   = _recoverNetwork.convergenceError;
            _fOfAlpha           = _recoverNetwork.fOfAlpha;
            _limitIteration    = _recoverNetwork.limitIteration;
            _trainingIteration = _recoverNetwork.trainingIteration;
            [self removeNetwork];
            _trainedNetwork     = _recoverNetwork;
            [NSUserDefaults saveTrainedNetwork:_trainedNetwork forKey:_kTrainedNetworkInfo];
        });
    }
}

/*
 * @ Recovers saved trained-network of ANN
 *   - 復原已儲存的訓練過的 ANN Network 數據資料
 */
-(void)recoverNetwork
{
    [self recoverNetwork:self.trainedNetwork];
}

#pragma --mark Blocks
-(void)setTrainingCompletion:(KRANNTrainingCompletion)_theBlock
{
    _trainingCompletion = _theBlock;
}

-(void)setPerIteration:(KRANNPerIteration)_theBlock
{
    _perIteration     = _theBlock;
}

#pragma --mark Getters
-(NSDictionary *)trainedInfo
{
    //代表有不落在 -1.0 ~ 1.0 之間的值，就回復至原先的數值長度與型態
    if( self._maxMultiple != 1 )
    {
        NSMutableArray *_formatedOutputResults = [NSMutableArray new];
        for( NSNumber *_result in _outputResults )
        {
            //還原每一個 goalValue 當初設定的原同等位同寬度的結果值，即返回原值域
            double _recoveredRetuls = [_result doubleValue] * self._maxMultiple;
            [_formatedOutputResults addObject:[NSNumber numberWithDouble:_recoveredRetuls]];
        }
        self.outputResults     = nil;
        _outputResults         = [[NSArray alloc] initWithArray:_formatedOutputResults];
        [_formatedOutputResults removeAllObjects];
        _formatedOutputResults = nil;
    }
    
    return @{KRANNTrainedInputWeights   : _inputWeights,
             KRANNTrainedHiddenLayers   : _hiddenLayers,
             KRANNTrainedHiddenWeights  : _allHiddenWeights,
             KRANNTrainedHiddenBiases   : _allHiddenBiases,
             KRANNTrainedOutputBiases   : _outputBiases,
             KRANNTrainedOutputResults  : _outputResults,
             KRANNTrainedIterations    : [NSNumber numberWithInteger:_trainingIteration]};
}

-(KRANNTrainedNetwork *)trainedNetwork
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


