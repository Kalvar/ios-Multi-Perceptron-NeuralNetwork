//
//  KRMLPNetworkOutput.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/7/9.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRMLPResult : NSObject

@property (nonatomic, assign) NSInteger outputIndex;
@property (nonatomic, assign) double outputValue;
@property (nonatomic, readonly) NSString *probability; // The percent of probability.

- (instancetype)init;

@end
