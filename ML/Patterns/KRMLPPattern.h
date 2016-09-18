//
//  KRMLPPattern.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/6/19.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRMLPPattern : NSObject

@property (nonatomic, strong) NSMutableArray <NSNumber *> *features;
@property (nonatomic, strong) NSMutableArray <NSNumber *> *targets;

- (instancetype)initWithFeatures:(NSArray <NSNumber *> *)f targets:(NSArray <NSNumber *> *)t;
- (instancetype)init;

- (void)addFeature:(NSNumber *)aFeature;
- (void)addFeaturesFromArray:(NSArray *)theFeatures;
- (void)addTarget:(NSNumber *)aTarget;
- (void)addTargetsFromArray:(NSArray *)theTargets;

@end
