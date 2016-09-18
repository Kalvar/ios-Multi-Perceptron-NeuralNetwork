//
//  KRMLPOutputNet.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/5/2.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPNet.h"

@interface KRMLPOutputNet : KRMLPNet

@property (nonatomic, assign) double targetValue; // 神經元期望輸出值

- (instancetype)init;

@end
