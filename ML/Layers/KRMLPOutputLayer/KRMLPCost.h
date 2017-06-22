//
//  KRMLPOutput.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/7/10.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRMLPCost : NSObject

@property (nonatomic, readonly) NSInteger patternsCount; // Patterns counting number.
@property (nonatomic, readonly) NSInteger outputsCount;  // Output nets counting number.
@property (nonatomic, readonly) double mse;              // The costValue / (patternsCount * outputsCount) / 2.0f
@property (nonatomic, readonly) double rmse;             // The costValue / (patternsCount * outputsCount)
@property (nonatomic, readonly) double crossEntropy;     // Not all situations can use Cross-Entropy to optimize, in some cases the MSE better than it.

- (instancetype)init;

- (void)addOutputs:(NSArray <NSNumber *> *)outputs targets:(NSArray <NSNumber *> *)targets;
- (void)removeRecords;

@end
