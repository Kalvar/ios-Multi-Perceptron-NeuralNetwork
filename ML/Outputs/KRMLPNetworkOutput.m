//
//  KRMLPNetworkOutput.m
//  KRMLP
//
//  Created by Kalvar Lin on 2016/8/21.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPNetworkOutput.h"

@implementation KRMLPNetworkOutput

- (instancetype)init
{
    self = [super init];
    if( self )
    {
        _results          = [NSMutableArray new];
        _currentIteration = 0;
    }
    return self;
}

- (void)buildResultsWithOutputs:(NSArray<NSNumber *> *)outputs
{
    [self removeAllResults];
    __weak typeof(self) weakSelf = self;
    [outputs enumerateObjectsUsingBlock:^(NSNumber * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        __strong typeof(weakSelf) strongSelf = weakSelf;
        KRMLPResult *result = [[KRMLPResult alloc] init];
        result.outputIndex  = idx;
        result.outputValue  = [obj doubleValue];
        [strongSelf.results addObject:result];
    }];
}

- (void)removeAllResults
{
    if( nil != _results )
    {
        [_results removeAllObjects];
    }
}

@end
