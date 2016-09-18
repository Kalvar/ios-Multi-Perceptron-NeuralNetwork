//
//  ViewController.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/4/27.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "ViewController.h"

#import "KRMLP.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    [self test];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)test
{
    // Creating patterns
    // Identifying the numbers (0 ~ 9), and verifying the 7.
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
    
    NSMutableArray <KRMLPPattern *> *patterns = [NSMutableArray new];
    [features enumerateObjectsUsingBlock:^(id  _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        KRMLPPattern *pattern = [[KRMLPPattern alloc] initWithFeatures:obj targets:[targets objectAtIndex:idx]];
        [patterns addObject:pattern];
    }];
    
    // MLP
    KRMLP *mlp            = [[KRMLP alloc] init];
    mlp.maxIteration      = 30;
    mlp.convergenceError  = 0.001f;
    mlp.networkActivation = KRMLPNetActivationSigmoid;
    
    mlp.initialMaxWeight  = 0.5f;
    mlp.initialMinWeight  = -0.5f;
    mlp.initialOptimize   = YES;
    
    //[mlp recoverForKey:@"A1"];
    [mlp addPatternsFromArray:patterns];
    
    [mlp setupOptimizationMethod:KRMLPOptimizationFixedInertia inertialRate:0.7f];
    
    KRMLPHiddenLayer *hiddenLayer1 = [mlp createHiddenLayerWithAutomaticSetting]; // [mlp createHiddenLayerWithNetCount:18 inputCount:36];
    [mlp addHiddenLayer:hiddenLayer1];
    
    KRMLPHiddenLayer *hiddenLayer2 = [mlp createHiddenLayerWithAutomaticSetting]; //[mlp createHiddenLayerDependsOnHiddenLayer:hiddenLayer1 netCount:18];
    [mlp addHiddenLayer:hiddenLayer2];

    KRMLPHiddenLayer *hiddenLayer3 = [mlp createHiddenLayerWithAutomaticSetting]; //[mlp createHiddenLayerDependsOnHiddenLayer:hiddenLayer2 netCount:16];
    [mlp addHiddenLayer:hiddenLayer3];
    
    [mlp setupOutputLayer];
    
    [mlp trainingWithCompletion:^(KRMLP *network) {
        
        //[network saveForKey:@"A1"];
        
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
        //NSLog(@"Training outputs of pattern[%li] : %@", trainingOutput.patternIndex, trainingOutput.outputs);
    }];
    
}

@end
