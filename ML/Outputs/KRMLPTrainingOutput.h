//
//  KRMLPTrainingOutput.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/8/22.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRMLPTrainingOutput : NSObject

@property (nonatomic, assign) NSInteger patternIndex;
@property (nonatomic, assign) NSArray <NSNumber *> *outputs;

- (instancetype)initWithPatternIndex:(NSInteger)patternIndex outputs:(NSArray <NSNumber *> *)outputs;

@end
