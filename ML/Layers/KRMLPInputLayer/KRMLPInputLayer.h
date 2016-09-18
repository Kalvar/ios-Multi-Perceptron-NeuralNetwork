//
//  KRMLPInputLayer.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/6/19.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRMLPPattern;

@interface KRMLPInputLayer : NSObject

@property (nonatomic, strong) NSMutableArray <KRMLPPattern *> *nets;
@property (nonatomic, readonly) NSInteger inputNetsCount; // To count how many input nets.

- (instancetype)init;
- (void)addNet:(KRMLPPattern *)net;

@end
