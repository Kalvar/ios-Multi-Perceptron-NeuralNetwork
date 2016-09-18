//
//  KRMLPLayer.h
//  KRMLP
//
//  Created by Kalvar Lin on 2016/8/22.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

@import Foundation;

@interface KRMLPLayer : NSObject <NSCopying, NSCoding>

@end

@interface KRMLPLayer (NSCoding)

- (void)encodeObject:(id)object forKey:(NSString *)key;
- (id)decodeForKey:(NSString *)key;

@end
