//
//  KRRBFFetcher.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/4/14.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRMLPPassed.h"

@interface KRMLPFetcher : NSObject

+ (instancetype)sharedFetcher;
- (instancetype)init;

- (void)save:(KRMLPPassed *)object forKey:(NSString *)key;  // Saving network information with key.
- (void)removeForKey:(NSString *)key;                       // Removes saved network information with key.
- (KRMLPPassed *)objectForKey:(NSString *)key;              // Fetching saved network with key.

@end
