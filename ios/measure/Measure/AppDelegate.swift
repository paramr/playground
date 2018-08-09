//
//  AppDelegate.swift
//  Measure
//
//  Created by Param Reddy on 8/8/18.
//  Copyright © 2018 spinorX. All rights reserved.
//

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

  private var mainWindow_: UIWindow!


  func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
    let viewController = ViewController()
    mainWindow_ = UIWindow(frame: UIScreen.main.bounds)
    mainWindow_.rootViewController = viewController
    mainWindow_.makeKeyAndVisible()

    return true
  }
}

