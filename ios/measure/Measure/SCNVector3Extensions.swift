//
//  ViewController.swift
//  Measure
//
//  Created by Param Reddy on 8/8/18.
//  Copyright © 2018 spinorX. All rights reserved.
//

import Foundation
import SceneKit

extension SCNVector3 {
  func length() -> Float {
    return sqrtf(x*x + y*y + z*z)
  }

  func distance(to: SCNVector3) -> Float {
    return (self - to).length()
  }

  static func midpointBetween(vector1: SCNVector3, vector2: SCNVector3) -> SCNVector3 {
    return SCNVector3Make((vector1.x + vector2.x) / 2, (vector1.y + vector2.y) / 2, (vector1.z + vector2.z) / 2)
  }

  static func eulerAngles(vector: SCNVector3) -> SCNVector3 {
    let height = vector.length()
    let lxz = sqrtf(vector.x * vector.x + vector.z * vector.z)
    let pitchB = vector.y < 0 ? Float.pi - asinf(lxz/height) : asinf(lxz/height)
    let pitch = vector.z == 0 ? pitchB : sign(vector.z) * pitchB

    var yaw: Float = 0
    if vector.x != 0 || vector.z != 0 {
      let inner = vector.x / (height * sinf(pitch))
      if inner > 1 || inner < -1 {
        yaw = Float.pi / 2
      } else {
        yaw = asinf(inner)
      }
    }
    return SCNVector3(CGFloat(pitch), CGFloat(yaw), 0)
  }
}

func + (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
  return SCNVector3Make(left.x + right.x, left.y + right.y, left.z + right.z)
}

func - (left: SCNVector3, right: SCNVector3) -> SCNVector3 {
  return SCNVector3Make(left.x - right.x, left.y - right.y, left.z - right.z)
}
