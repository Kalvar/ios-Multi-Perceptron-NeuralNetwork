//
//  KRMLPTrainingOutput.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/8/22.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPTrainingOutput.h"

@implementation KRMLPTrainingOutput

- (instancetype)initWithPatternIndex:(NSInteger)patternIndex outputs:(NSArray<NSNumber *> *)outputs
{
    self = [super init];
    if( self )
    {
        _patternIndex = patternIndex;
        _outputs      = outputs;
    }
    return self;
}

@end
