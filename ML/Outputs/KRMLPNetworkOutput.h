//
//  KRMLPNetworkOutput.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/8/21.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPResult.h"

@interface KRMLPNetworkOutput : NSObject

@property (nonatomic, strong) NSMutableArray <KRMLPResult *> *results;
@property (nonatomic, assign) NSInteger currentIteration; // 目前的迭代

- (void)buildResultsWithOutputs:(NSArray <NSNumber *> *)outputs;
- (void)removeAllResults;

@end
