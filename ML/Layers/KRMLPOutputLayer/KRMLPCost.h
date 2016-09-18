//
//  KRMLPOutput.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/7/10.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRMLPCost : NSObject

@property (nonatomic, assign) double costValue;        // The (targetValue - outputValue)^2.
@property (nonatomic, assign) NSInteger patternsCount; // Patterns counting number.
@property (nonatomic, assign) NSInteger outputsCount;  // Output nets counting number.
@property (nonatomic, readonly) double mse;            // The costValue / (patternsCount * outputsCount) / 2.0f
@property (nonatomic, readonly) double rmse;           // The costValue / (patternsCount * outputsCount)

- (instancetype)init;

@end
