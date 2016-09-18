//
//  KRMLP.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/4/27.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLP.h"
#import "KRMLPCost.h"
#import "KRMLPFetcher.h"

@interface KRMLP ()

@property (nonatomic, strong) KRMLPCost *cost;
@property (nonatomic, assign) NSInteger currentIteration;
@property (nonatomic, copy) KRMLPNetworkOutputBlock networkOutputBlock;
@property (nonatomic, copy) KRMLPIterationBlock iterationBlock;
@property (nonatomic, copy) KRMLPTrainingOutputBlock trainingOutputBlock;

@end

@implementation KRMLP (Configs)

- (void)setupNetworkActivation
{
    KRMLPNetActivations networkActivation = self.networkActivation;
    if( networkActivation != KRMLPNetActivationDefault )
    {
        if( self.hiddenLayers )
        {
            [self.hiddenLayers enumerateObjectsUsingBlock:^(KRMLPHiddenLayer * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
                obj.activeFunction = networkActivation;
            }];
        }
        
        if( self.outputLayer )
        {
            self.outputLayer.activeFunction = networkActivation;
        }
    }
}

- (void)setupCostFunction
{
    KRMLPCost *cost    = self.cost;
    cost.patternsCount = self.patternsCount;
    cost.outputsCount  = self.outputNetsCount;
}

- (void)initializeSettings
{
    self.currentIteration = 0;
}

- (double)optimizeWeightWithValue:(double)someValue netCount:(NSInteger)netCount
{
    if( YES == self.initialOptimize )
    {
        // 優化權重初始規則 : (someValue / 此層神經元個數)
        return ( someValue / netCount );
    }
    return someValue;
}

@end

@implementation KRMLP (Training)

// The all outputs of hidden layers.
- (NSMutableArray <NSArray *> *)hiddenLayersOutputsWithInputs:(NSArray <NSNumber *> *)inputs
{
    // Calculating the outputs of Hidden Layers, and :
    //   - layersOutputs will include inputLayer outputs and hidden layers outputs,
    //   - layersOutputs will be used in updating weights and biases, too.
    __block NSMutableArray <NSArray *> *layersOutputs = [[NSMutableArray alloc] initWithObjects:inputs, nil];
    NSArray <KRMLPHiddenLayer *> *hiddenLayers        = self.hiddenLayers;
    [hiddenLayers enumerateObjectsUsingBlock:^(KRMLPHiddenLayer * _Nonnull hiddenLayer, NSUInteger idx, BOOL * _Nonnull stop) {
        // Current layer outputs are next layer inputs, the last layer outputs ( [layersOutputs lastObject] ) are current layer inputs the same.
        NSArray <NSNumber *> *nextInputs = [hiddenLayer layerOutputsWithInputs:[layersOutputs lastObject]];
        // To add the current layer outputs to be next layer inputs with layersOutputs array.
        [layersOutputs addObject:nextInputs];
    }];
    return layersOutputs;
}

- (void)trainingWithInputs:(NSArray <NSNumber *> *)inputs targets:(NSArray <NSNumber *> *)targets
{
    NSArray <KRMLPHiddenLayer *> *hiddenLayers        = self.hiddenLayers;
    __block NSMutableArray <NSArray *> *layersOutputs = [self hiddenLayersOutputsWithInputs:inputs];
    
    // Network final outputs of this pattern.
    KRMLPOutputLayer *outputLayer = self.outputLayer;
    [outputLayer layerOutputsWithInputs:[layersOutputs lastObject]];
    
    // 多分類的 output layer to hidden layer 跟 hidden layer to hidden layer / input layer 的 delta value (誤差值) 算法都一樣。
    // Recording cost value, btw, the delta-values of output-layer-nets are recorded in the output-nets self (net.deltaValue).
    KRMLPCost *cost = self.cost;
    cost.costValue  = [outputLayer calculateCostAndDeltaWithTargets:targets];
    
    // Reversing and looping hiddenLayers to start in calculating the delta values from output-layer to hidden layers (that includes hidden layer to hidden layer).
    // lastDeltas the initial value is delta-values of output-nets, the delta-value comes from net.deltaValue.
    NSMutableArray <NSArray <KRMLPNet *> *> *lastDeltas = [[NSMutableArray alloc] initWithObjects:outputLayer.nets, nil];
    [hiddenLayers enumerateObjectsWithOptions:NSEnumerationReverse usingBlock:^(KRMLPHiddenLayer * _Nonnull hiddenLayer, NSUInteger idx, BOOL * _Nonnull stop) {
        NSArray <KRMLPNet *> *currentNets = [hiddenLayer calculateDeltasWithLastLayerDeltas:[lastDeltas lastObject]];
        // 裡面會是 " 輸出層一路逆回輸入層 " 的順序, lastDeltas 的最後 1 個 Object 是第 1 層的 Hidden Layer,
        // 而 lastDeltas 純粹是用來運算每一層每顆神經元的 Delta-Value 而已。
        [lastDeltas addObject:currentNets];
    }];
    
    // 計算完 Output Layer 的 Delta Values 和 Hidden Layers 的 Delta Values 後，就能開始修正權重與偏權，
    // Input Layer 沒有權重與偏權，只有單純的輸入向量，權重和偏權值都是掛在下一層的 Hidden Layer / Output Layer Nets 身上。
    
    // 開始往回修正權重和偏權, One pattern to update once.
    // 先修正 Output Layer, 需要最後一個的隱藏層輸出值一起進行運算。
    [outputLayer updateWithHiddenLayerOutputs:[layersOutputs lastObject] learningRate:self.learningRate];
    [layersOutputs removeLastObject];
    
    // 再修正 Last Hidden Layer (index last) to First Hidden Layers (index 0),
    // 而修正第 1 層的 Hidden Layer 時，需要 Input Layer 的 Inputs (patterns) 來做運算。
    // 逆取 Hidden Layers 出來做修正。
    __weak typeof(self) weakSelf = self;
    [hiddenLayers enumerateObjectsWithOptions:NSEnumerationReverse usingBlock:^(KRMLPHiddenLayer * _Nonnull hiddenLayer, NSUInteger layerIndex, BOOL * _Nonnull stop) {
        __strong typeof(self) strongSelf = weakSelf;
        [hiddenLayer updateWithLastLayerOutputs:[layersOutputs lastObject] learningRate:strongSelf.learningRate];
        [layersOutputs removeLastObject];
    }];
    
    // Free the memory in last step.
    [lastDeltas removeAllObjects];
}

- (void)iterationTraining
{
    self.currentIteration += 1;
    
    // The self.patterns is getter of inputLayer.nets.
    NSInteger patternIndex = -1;
    for( KRMLPPattern *pattern in self.patterns )
    {
        patternIndex += 1;
        [self trainingWithInputs:pattern.features targets:pattern.targets];
        if( self.trainingOutputBlock )
        {
            KRMLPTrainingOutput *trainingOutput = [[KRMLPTrainingOutput alloc] initWithPatternIndex:patternIndex
                                                                                            outputs:self.outputLayer.outputs];
            self.trainingOutputBlock(trainingOutput);
        }
    }
    
    if( self.cost.rmse <= self.convergenceError || self.currentIteration >= self.maxIteration )
    {
        if( self.networkOutputBlock )
        {
            self.networkOutputBlock(self);
        }
    }
    else
    {
        if( self.iterationBlock )
        {
            self.iterationBlock(self);
        }
        [self iterationTraining];
    }
}

@end

@implementation KRMLP

+ (instancetype)sharedNetwork
{
    static dispatch_once_t pred;
    static KRMLP *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRMLP alloc] init];
    });
    return _object;
}

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _inputLayer        = [[KRMLPInputLayer alloc] init];
        _hiddenLayers      = [NSMutableArray new];
        _outputLayer       = [[KRMLPOutputLayer alloc] init];
        
        _learningRate      = 0.8f;
        _networkActivation = KRMLPNetActivationDefault;
        
        _initialMaxWeight  = 0.5f;
        _initialMinWeight  = -0.5f;
        
        _convergenceError  = 0.001f;
        
        _cost              = [[KRMLPCost alloc] init];
    }
    return self;
}

#pragma mark - Calculating
- (NSInteger)calculateLayerNetCountWithInputCount:(NSInteger)inputCount outputCount:(NSInteger)outputCount
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
    NSInteger netCount = (NSInteger)powf(( inputCount * outputCount ), 0.5f);
    if( netCount < 1 )
    {
        netCount = 1;
    }
    return netCount;
}

#pragma mark - Creating
- (KRMLPPattern *)createPatternWithFeatures:(NSArray <NSNumber *> *)features targets:(NSArray <NSNumber *> *)targets
{
    return [[KRMLPPattern alloc] initWithFeatures:features targets:targets];
}

- (KRMLPHiddenLayer *)createHiddenLayerWithIndex:(NSNumber *)layerIndex nets:(NSArray <KRMLPHiddenNet *> *)nets
{
    return [[KRMLPHiddenLayer alloc] initWithIndex:layerIndex nets:nets];
}

// 建立 Hidden Layer, 並建立 netCount 個 nets，且其 nets 的隨機權重在 minWeight ~ maxWeight 之間
// netCount             : 神經元數目
// inputCount           : 神經元的輸入特徵向量數量，用於預設連結的權重數量 ; 上一層有幾顆神經元，就有幾個傳進來的向量值
// maxWeight, minWeight : 神經元隨機權重範圍
- (KRMLPHiddenLayer *)createHiddenLayerWithNetCount:(NSInteger)netCount inputCount:(NSInteger)inputCount maxWeight:(double)maxWeight min:(double)minWeight
{
    // To optimize the max and min weights if needed.
    maxWeight                     = [self optimizeWeightWithValue:maxWeight netCount:netCount];
    minWeight                     = [self optimizeWeightWithValue:minWeight netCount:netCount];
    // To create the hidden layer.
    KRMLPHiddenLayer *hiddenLayer = [[KRMLPHiddenLayer alloc] init];
    hiddenLayer.index             = @(self.hiddenLayerCount);
    for( NSInteger i=0; i<netCount; i++ )
    {
        KRMLPHiddenNet *net = [[KRMLPHiddenNet alloc] init];
        [net randomizeWeightsAtCount:inputCount max:maxWeight min:minWeight];
        [hiddenLayer addNet:net];
    }
    return hiddenLayer;
}

// The inputCount comes from how many inputs of last connected layer (hidden layer or input layer).
- (KRMLPHiddenLayer *)createHiddenLayerWithNetCount:(NSInteger)netCount inputCount:(NSInteger)inputCount
{
    return [self createHiddenLayerWithNetCount:netCount inputCount:inputCount maxWeight:_initialMaxWeight min:_initialMinWeight];
}

// Automatic setup the nets count.
- (KRMLPHiddenLayer *)createHiddenLayerWithAutomaticSetting
{
    // 已經有上一層的 Hidden Layer 存在 ? 取出上一層的 Hidden Layer Nets Count 當 Input Nets Count : 取出 Input Layer Nets Count.
    NSInteger inputNetsCount = ( self.hiddenLayerCount > 0 )? [[_hiddenLayers lastObject].nets count] : self.inputNetsCount;
    // 計算這 Hidden Layer 需要幾顆神經元
    NSInteger netCount       = [self calculateLayerNetCountWithInputCount:inputNetsCount outputCount:self.outputNetsCount];
    return [self createHiddenLayerWithNetCount:netCount inputCount:inputNetsCount maxWeight:_initialMaxWeight min:_initialMinWeight];
}

// dependentLayer means the nets number of new hidden layer is depend on dependentLayer.
- (KRMLPHiddenLayer *)createHiddenLayerDependsOnHiddenLayer:(KRMLPHiddenLayer *)dependentLayer netCount:(NSInteger)netCount maxWeight:(double)maxWeight min:(double)minWeight
{
    return [self createHiddenLayerWithNetCount:netCount inputCount:[dependentLayer.nets count] maxWeight:maxWeight min:minWeight];
}

- (KRMLPHiddenLayer *)createHiddenLayerDependsOnHiddenLayer:(KRMLPHiddenLayer *)dependentLayer netCount:(NSInteger)netCount
{
    return [self createHiddenLayerDependsOnHiddenLayer:dependentLayer netCount:netCount maxWeight:_initialMaxWeight min:_initialMinWeight];
}

#pragma mark - Setups
- (void)addPattern:(KRMLPPattern *)pattern
{
    if( nil != pattern )
    {
        [_inputLayer addNet:pattern];
    }
}

- (void)addPatternsFromArray:(NSArray <KRMLPPattern *> *)samples
{
    if( nil != samples && [samples count] > 0 )
    {
        for( KRMLPPattern *pattern in samples )
        {
            [self addPattern:pattern];
        }
    }
}

- (void)addHiddenLayer:(KRMLPHiddenLayer *)hiddenLayer
{
    if( nil != hiddenLayer )
    {
        [_hiddenLayers addObject:hiddenLayer];
    }
}

- (void)addHiddenLayer:(KRMLPHiddenLayer *)hiddenLayer dependsOnHiddenLayer:(KRMLPHiddenLayer *)dependentLayer
{
    // 把該 Hidden Layer 插入要跟它連動的 dependentLayer 後面
    NSInteger dependenceIndex = [_hiddenLayers indexOfObject:dependentLayer];
    if( NSNotFound == dependenceIndex )
    {
        dependenceIndex = -1;
    }
    dependenceIndex += 1;
    [_hiddenLayers insertObject:hiddenLayer atIndex:dependenceIndex];
}

// Adding a net to specific hidden layer.
- (void)addHiddenNet:(KRMLPHiddenNet *)hiddenNet forLayerIndex:(NSInteger)layerIndex
{
    if( layerIndex >= self.hiddenLayerCount )
    {
        return;
    }
    
    KRMLPHiddenLayer *hiddenLayer = [_hiddenLayers objectAtIndex:layerIndex];
    [hiddenLayer addNet:hiddenNet];
}

- (void)addOutpuNet:(KRMLPOutputNet *)outputNet
{
    [_outputLayer addNet:outputNet];
}

- (void)setupOutputLayerWithNetMaxWeight:(double)maxWeight min:(double)minWeight
{
    // Depends on the last hidden layer.
    KRMLPHiddenLayer *dependentLayer = [_hiddenLayers lastObject];
    // netCount is the counting number of targets of pattern. 有幾個輸出目標，就有幾個 Output Net.
    NSInteger netCount               = self.outputNetsCount;
    NSInteger inputCount             = [dependentLayer.nets count];
    // To optimize the max, min weights if needed.
    maxWeight                        = [self optimizeWeightWithValue:maxWeight netCount:netCount];
    minWeight                        = [self optimizeWeightWithValue:minWeight netCount:netCount];
    // Start in create nets of output layer.
    for(NSInteger i=0; i<netCount; i++)
    {
        KRMLPOutputNet *net = [[KRMLPOutputNet alloc] init];
        [net randomizeWeightsAtCount:inputCount max:maxWeight min:minWeight];
        [_outputLayer addNet:net];
    }
}

- (void)setupOutputLayer
{
    [self setupOutputLayerWithNetMaxWeight:_initialMaxWeight min:_initialMinWeight];
}

#pragma mark - Training
- (void)trainingWithCompletion:(KRMLPNetworkOutputBlock)completion iteration:(KRMLPIterationBlock)iteration training:(KRMLPTrainingOutputBlock)training
{
    _networkOutputBlock  = completion;
    _iterationBlock      = iteration;
    _trainingOutputBlock = training;
    
    [self initializeSettings];
    
    // If self.networkActivation keeps KRMLPNetActivationDefault that means we wanna setup the active functions of hidden layears and output layer by ourself.
    [self setupNetworkActivation];
    
    // Setups cost function.
    [self setupCostFunction];
    
    // Start in iteration.
    [self iterationTraining];
}

- (void)trainingWithCompletion:(KRMLPNetworkOutputBlock)completion iteration:(KRMLPIterationBlock)iteration
{
    [self trainingWithCompletion:completion iteration:iteration training:nil];
}

- (void)training
{
    [self trainingWithCompletion:nil iteration:nil];
}

#pragma mark - Predication
- (void)predicateWithFeatures:(NSArray <NSNumber *> *)features completion:(KRMLPNetworkPredicationBlock)completion
{
    if( completion )
    {
        NSMutableArray <NSArray *> *layersOutputs = [self hiddenLayersOutputsWithInputs:features];
        NSArray <NSNumber *> *outputs             = [_outputLayer directOutputsWithInputs:[layersOutputs lastObject]];
        
        KRMLPNetworkOutput *networkOutput         = [[KRMLPNetworkOutput alloc] init];
        networkOutput.currentIteration            = 1;
        [networkOutput buildResultsWithOutputs:outputs];
        
        completion(networkOutput);
    }
}

#pragma mark - Saver & Recover
- (void)saveForKey:(NSString *)key
{
    KRMLPPassed *passed      = [[KRMLPPassed alloc] init];
    passed.hiddenLayers      = [_hiddenLayers copy];
    passed.outputLayer       = [_outputLayer copy];
    passed.networkActivation = _networkActivation;
    passed.learningRate      = _learningRate;
    
    KRMLPFetcher *fetcher    = [[KRMLPFetcher alloc] init];
    [fetcher save:passed forKey:key];
}

- (void)recoverForKey:(NSString *)key
{
    KRMLPFetcher *fetcher = [[KRMLPFetcher alloc] init];
    KRMLPPassed *passed   = [fetcher objectForKey:key];
    if( nil != passed )
    {
        _hiddenLayers      = passed.hiddenLayers;
        _outputLayer       = passed.outputLayer;
        _networkActivation = passed.networkActivation;
        _learningRate      = passed.learningRate;
    }
}

#pragma mark - Setup Optimization
- (void)setupOptimizationMethod:(KRMLPOptimizationMethods)method inertialRate:(double)inertialRate
{
    // 必須照順序先設定 Method -> 再設定 Inertial Rate
    KRMLPOptimization *optimization = [KRMLPOptimization shared];
    optimization.method             = method;
    optimization.inertialRate       = inertialRate;
}

- (void)setupOptimizationMethod:(KRMLPOptimizationMethods)method
{
    KRMLPOptimization *optimization = [KRMLPOptimization shared];
    optimization.method             = method;
}

- (void)setupOptimizationInertialRate:(double)inertialRate
{
    KRMLPOptimization *optimization = [KRMLPOptimization shared];
    optimization.inertialRate       = inertialRate;
}

#pragma mark - Getters
- (NSMutableArray <KRMLPPattern *> *)patterns
{
    return ( nil != _inputLayer ) ? _inputLayer.nets : nil;
}

- (NSInteger)patternsCount
{
    NSArray *patterns = self.patterns;
    return ( nil != patterns ) ? [patterns count] : 0;
}

- (NSInteger)hiddenLayerCount
{
    return ( nil != _hiddenLayers ) ? [_hiddenLayers count] : 0;
}

- (NSInteger)outputNetsCount
{
    KRMLPPattern *pattern = [self.patterns firstObject];
    if( pattern && pattern.targets )
    {
        return [pattern.targets count];
    }
    return 0;
}

- (NSInteger)inputNetsCount
{
    return ( nil != _inputLayer )? _inputLayer.inputNetsCount : 0;
}

- (NSInteger)iteration
{
    return _currentIteration;
}

@end
