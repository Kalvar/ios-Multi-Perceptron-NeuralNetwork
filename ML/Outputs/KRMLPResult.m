//
//  KRMLPNetworkOutput.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/7/9.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPResult.h"

@implementation KRMLPResult

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _outputIndex = 0;
        _outputValue = 0;
    }
    return self;
}

#pragma mark - Getters
- (NSString *)probability
{
    return [NSString stringWithFormat:@"%.2f", (_outputValue * 100.0f)];
}

@end
