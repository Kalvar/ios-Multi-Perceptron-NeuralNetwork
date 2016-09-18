ios-Multi-Perceptron-NeuralNetwork
=================

Machine Learning (マシンラーニング) in this project, it implemented multi-layer perceptrons neural network (MLP) (ニューラルネットワーク) and Back Propagation Neural Network (BPN). It designed unlimited hidden layers to do the training tasks. This network can be used in products recommendation (おすすめの商品), user behavior analysis (ユーザーの行動分析), data mining (データマイニング) and data analysis (データ分析).

#### Podfile

```ruby
platform :ios, '7.0'
pod "KRMLP", "~> 2.2.0"
```

## How to use

#### Import

``` objective-c
#import "KRMLP.h"
```

#### Creating training samples

Using KRMLPPattern to create the training patterns.
``` objective-c
KRMLPPattern *pattern = [[KRMLPPattern alloc] initWithFeatures:@[@1, @0, @0.5f] targets:@[@0, @1]];
```

For example, these training samples are going to identify 0 to 9 numbers, it is going to create the features (4 x 9) of patterns in first step.

``` objective-c
NSArray *features = @[// 0
                      @[@1, @1, @1, @1,
                        @1, @1, @1, @1,
                        @1, @1, @0, @0,
                        @0, @0, @0, @0,
                        @0, @1, @1, @0,
                        @0, @0, @0, @0,
                        @0, @0, @1, @1,
                        @1, @1, @1, @1,
                        @1, @1, @1, @1],
                      
                      // 1
                      @[@0, @0, @0, @0,
                        @0, @0, @0, @0,
                        @0, @0, @0, @0,
                        @0, @0, @0, @0,
                        @0, @0, @0, @0,
                        @0, @0, @0, @0,
                        @0, @0, @0, @1,
                        @1, @1, @1, @1,
                        @1, @1, @1, @1],
                      
                      // 2
                      @[@1, @0, @0, @0,
                        @1, @1, @1, @1,
                        @1, @1, @0, @0,
                        @0, @1, @0, @0,
                        @0, @1, @1, @0,
                        @0, @0, @1, @0,
                        @0, @0, @1, @1,
                        @1, @1, @1, @1,
                        @0, @0, @0, @1],
                      
                      // 3
                      @[@1, @0, @0, @0,
                        @1, @0, @0, @0,
                        @1, @1, @0, @0,
                        @0, @1, @0, @0,
                        @0, @1, @1, @0,
                        @0, @0, @1, @0,
                        @0, @0, @1, @1,
                        @1, @1, @1, @1,
                        @1, @1, @1, @1],
                      
                      // 4
                      @[@1, @1, @1, @1,
                        @1, @0, @0, @0,
                        @0, @0, @0, @0,
                        @0, @1, @0, @0,
                        @0, @0, @0, @0,
                        @0, @0, @1, @0,
                        @0, @0, @0, @1,
                        @1, @1, @1, @1,
                        @1, @1, @1, @1],
                      
                      // 5
                      @[@1, @1, @1, @1,
                        @1, @0, @0, @0,
                        @1, @1, @0, @0,
                        @0, @1, @0, @0,
                        @0, @1, @1, @0,
                        @0, @0, @1, @0,
                        @0, @0, @1, @1,
                        @0, @0, @0, @1,
                        @1, @1, @1, @1],
                      
                      // 6
                      @[@1, @1, @1, @1,
                        @1, @1, @1, @1,
                        @1, @1, @0, @0,
                        @0, @1, @0, @0,
                        @0, @1, @1, @0,
                        @0, @0, @1, @0,
                        @0, @0, @1, @1,
                        @0, @0, @0, @1,
                        @1, @1, @1, @1],
                      
                      // 7
                      @[@1, @0, @0, @0,
                        @0, @0, @0, @0,
                        @0, @1, @0, @0,
                        @0, @0, @0, @0,
                        @0, @0, @1, @0,
                        @0, @0, @0, @0,
                        @0, @0, @0, @1,
                        @1, @1, @1, @1,
                        @1, @1, @1, @1],
                      
                      // 8
                      @[@1, @1, @1, @1,
                        @1, @1, @1, @1,
                        @1, @1, @0, @0,
                        @0, @1, @0, @0,
                        @0, @1, @1, @0,
                        @0, @0, @1, @0,
                        @0, @0, @1, @1,
                        @1, @1, @1, @1,
                        @1, @1, @1, @1],
                      
                      // 9
                      @[@1, @1, @1, @1,
                        @1, @0, @0, @0,
                        @0, @1, @0, @0,
                        @0, @1, @0, @0,
                        @0, @0, @1, @0,
                        @0, @0, @1, @0,
                        @0, @0, @0, @1,
                        @1, @1, @1, @1,
                        @1, @1, @1, @1]
                      ];
```

Second step, to go to define the 10 targets (outputs) to map the features.
``` objective-c
NSArray *targets  = @[// 0
                      @[@1, @0, @0, @0, @0, @0, @0, @0, @0, @0],
                      // 1
                      @[@0, @1, @0, @0, @0, @0, @0, @0, @0, @0],
                      // 2
                      @[@0, @0, @1, @0, @0, @0, @0, @0, @0, @0],
                      // 3
                      @[@0, @0, @0, @1, @0, @0, @0, @0, @0, @0],
                      // 4
                      @[@0, @0, @0, @0, @1, @0, @0, @0, @0, @0],
                      // 5
                      @[@0, @0, @0, @0, @0, @1, @0, @0, @0, @0],
                      // 6
                      @[@0, @0, @0, @0, @0, @0, @1, @0, @0, @0],
                      // 7
                      @[@0, @0, @0, @0, @0, @0, @0, @1, @0, @0],
                      // 8
                      @[@0, @0, @0, @0, @0, @0, @0, @0, @1, @0],
                      // 9
                      @[@0, @0, @0, @0, @0, @0, @0, @0, @0, @1],
                      ];
```

Third step, to create the array of patterns.
``` objective-c
NSMutableArray <KRMLPPattern *> *patterns = [NSMutableArray new];
[features enumerateObjectsUsingBlock:^(id  _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
    KRMLPPattern *pattern = [[KRMLPPattern alloc] initWithFeatures:obj targets:[targets objectAtIndex:idx]];
    [patterns addObject:pattern];
}];
```

#### Training sample

Normal case:
``` objective-c
KRMLP *mlp            = [[KRMLP alloc] init];
mlp.maxIteration      = 300;
mlp.convergenceError  = 0.001f;
mlp.networkActivation = KRMLPNetActivationSigmoid;

mlp.initialMaxWeight  = 0.5f;
mlp.initialMinWeight  = -0.5f;
mlp.initialOptimize   = YES;

[mlp addPatternsFromArray:patterns];
[mlp setupOptimizationMethod:KRMLPOptimizationFixedInertia inertialRate:0.7f];

KRMLPHiddenLayer *hiddenLayer1 = [mlp createHiddenLayerWithAutomaticSetting]; // or [mlp createHiddenLayerWithNetCount:18 inputCount:36];
[mlp addHiddenLayer:hiddenLayer1];

KRMLPHiddenLayer *hiddenLayer2 = [mlp createHiddenLayerWithAutomaticSetting]; // or [mlp createHiddenLayerDependsOnHiddenLayer:hiddenLayer1 netCount:18];
[mlp addHiddenLayer:hiddenLayer2];

KRMLPHiddenLayer *hiddenLayer3 = [mlp createHiddenLayerWithAutomaticSetting]; // or [mlp createHiddenLayerDependsOnHiddenLayer:hiddenLayer2 netCount:16];
[mlp addHiddenLayer:hiddenLayer3];

[mlp setupOutputLayer];

[mlp trainingWithCompletion:^(KRMLP *network) {
    
    [network saveForKey:@"A1"];
    
    // Verifying #7 (the number is something wrong)
    NSArray *inputs = @[@1, @1, @1, @0,
                        @0, @0, @0, @0,
                        @0, @1, @0, @0,
                        @0, @0, @0, @0,
                        @0, @0, @1, @0,
                        @0, @0, @0, @0,
                        @0, @0, @0, @1,
                        @1, @1, @1, @1,
                        @1, @1, @1, @1];
    [network predicateWithFeatures:inputs completion:^(KRMLPNetworkOutput *networkOutput) {
        [networkOutput.results enumerateObjectsUsingBlock:^(KRMLPResult * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
            NSLog(@"Predicated the number [%li] is possible %@%%", obj.outputIndex, obj.probability);
        }];
    }];
} iteration:^(KRMLP *network) {
    NSLog(@"Iteration %li", network.iteration);
} training:^(KRMLPTrainingOutput *trainingOutput) {
    NSLog(@"Training outputs of pattern[%li] : %@", trainingOutput.patternIndex, trainingOutput.outputs);
}];
```

#### How to save / recover the trained network

To recover saved network that directly use [mlp recoverForKey:] to recall hiddenLayers, outputLayer, networkActivation and learningRate. Hence, you still could setup others parapmeters of KRMLP to train the new network or direct predicate something.

For example, to continually training.
``` objective-c
KRMLP *mlp            = [[KRMLP alloc] init];
[mlp recoverForKey:@"A1"];
mlp.maxIteration      = 50;
mlp.convergenceError  = 0.001f;
[mlp addPatternsFromArray:patterns];
[mlp trainingWithCompletion:^(KRMLP *network) {
    // If you want, to save to another record.
    [network saveForKey:@"B2"];
    // Do something since training finished.
    // ...
} iteration:^(KRMLP *network) {
    //NSLog(@"Iteration %li", network.iteration);
} training:^(KRMLPTrainingOutput *trainingOutput) {
    //NSLog(@"Training outputs of pattern[%li] : %@", trainingOutput.patternIndex, trainingOutput.outputs);
}];
```

If you wanna change any recalled parameters / settings that you could set it up after called [mlp recoverForKey:] method.
``` objective-c
KRMLP *mlp            = [[KRMLP alloc] init];
[mlp recoverForKey:@"A1"];
mlp.maxIteration      = 50;
mlp.convergenceError  = 0.001f;
mlp.networkActivation = KRMLPNetActivationSigmoid;
[mlp setupOptimizationMethod:KRMLPOptimizationQuickProp]; // Changes from KRMLPOptimizationFixedInertia to KRMLPOptimizationQuickProp.
// ...
```

#### Optimization Learning Methods

Fixed inertial rate:
``` objective-c
[mlp setupOptimizationMethod:KRMLPOptimizationFixedInertia inertialRate:0.7f];    
```

QuickProp:
``` objective-c
[mlp setupOptimizationMethod:KRMLPOptimizationQuickProp];
```

## Version

V2.2.0

## License

MIT.

## Todolist

1. RProp.
2. Mixes fixed inertia and QuickProp.
3. EDBD.
4. Protocol implementations.
