//
//  ViewController.swift
//  Measure
//
//  Created by Param Reddy on 8/8/18.
//  Copyright © 2018 spinorX. All rights reserved.
//

import UIKit
import ARKit

extension UIView {
  func constraintToFillParentView(_ parentView: UIView) {
    leftAnchor.constraint(equalTo: parentView.leftAnchor).isActive = true
    parentView.rightAnchor.constraint(equalTo: rightAnchor).isActive = true
    topAnchor.constraint(equalTo: parentView.topAnchor).isActive = true
    parentView.bottomAnchor.constraint(equalTo: bottomAnchor).isActive = true
  }

  func constraintToCenterInParentView(_ parentView: UIView) {
    centerXAnchor.constraint(equalTo: parentView.centerXAnchor).isActive = true
    centerYAnchor.constraint(equalTo: parentView.centerYAnchor).isActive = true
  }

  func setExactContainingSizeConstraints(priority: UILayoutPriority) {
    self.setContentCompressionResistancePriority(priority, for: .horizontal)
    self.setContentCompressionResistancePriority(priority, for: .vertical)
    self.setContentHuggingPriority(priority, for: .horizontal)
    self.setContentHuggingPriority(priority, for: .vertical)
    let widthConstaint = self.widthAnchor.constraint(equalToConstant: 0)
    widthConstaint.priority = .fittingSizeLevel
    widthConstaint.isActive = true
    let heightConstaint = self.heightAnchor.constraint(equalToConstant: 0)
    heightConstaint.priority = .fittingSizeLevel
    heightConstaint.isActive = true
  }
}

class ViewController: UIViewController, ARSCNViewDelegate {
  var sceneView_: ARSCNView!
  var measurementsLabel_: UILabel!
  var crossImageView_: UIImageView!
  var startNode_: SCNNode?
  var lineNode_: SCNNode?

  override func loadView() {
    sceneView_ = ARSCNView()
    sceneView_.delegate = self
    sceneView_.showsStatistics = true
    sceneView_.autoenablesDefaultLighting = true
    sceneView_.debugOptions = [ARSCNDebugOptions.showFeaturePoints]
    sceneView_.scene = SCNScene()

    measurementsLabel_ = UILabel()
    measurementsLabel_.backgroundColor = UIColor.black.withAlphaComponent(0.4)
    measurementsLabel_.textColor = UIColor.white
    measurementsLabel_.textAlignment = .center
    measurementsLabel_.numberOfLines = 0
    measurementsLabel_.lineBreakMode = .byWordWrapping

    crossImageView_ = UIImageView()
    crossImageView_.image = UIImage(named: "cross")!.withRenderingMode(.alwaysTemplate)
    crossImageView_.tintColor = UIColor.gray

    let rootView = UIView()

    rootView.addSubview(sceneView_)
    rootView.addSubview(measurementsLabel_)
    rootView.addSubview(crossImageView_)

    measurementsLabel_.translatesAutoresizingMaskIntoConstraints = false
    measurementsLabel_.setExactContainingSizeConstraints(priority: .required)
    measurementsLabel_.centerXAnchor.constraint(equalTo: rootView.safeAreaLayoutGuide.centerXAnchor).isActive = true
    measurementsLabel_.topAnchor.constraint(equalTo: rootView.safeAreaLayoutGuide.topAnchor, constant: 8).isActive = true
    crossImageView_.translatesAutoresizingMaskIntoConstraints = false
    crossImageView_.constraintToCenterInParentView(rootView)
    sceneView_.translatesAutoresizingMaskIntoConstraints = false
    sceneView_.constraintToFillParentView(rootView)

    let longPressRecognizer = UILongPressGestureRecognizer(
      target: self, action: #selector(handleLongTap(recognizer:)))
    longPressRecognizer.minimumPressDuration = 0.01
    sceneView_.addGestureRecognizer(longPressRecognizer)

    view = rootView
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    let config = ARWorldTrackingConfiguration()
    config.planeDetection = [.horizontal, .vertical]
    sceneView_.session.run(config)
  }

  override func viewWillDisappear(_ animated: Bool) {
    sceneView_.session.pause()
    super.viewWillAppear(animated)
  }

  func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
    DispatchQueue.main.async {
      guard let currentPosition = self.positionOnExistingPlanes() else {
        self.crossImageView_.tintColor = UIColor.gray
        return
      }
      self.crossImageView_.tintColor = UIColor.green
      guard let startNode = self.startNode_ else {
          return
      }
      self.lineNode_?.removeFromParentNode()
      self.lineNode_ = self.lineNode(from: startNode.position, to: currentPosition)
      self.sceneView_.scene.rootNode.addChildNode(self.lineNode_!)
      let distStr = self.distanceString(from: startNode.position, to: currentPosition)
      self.measurementsLabel_.text = distStr
    }
  }

  @objc
  private func handleLongTap(recognizer: UILongPressGestureRecognizer) {
    if recognizer.state == .began {
      if let position = positionOnExistingPlanes() {
        let node = nodeWithPosition(position)
        sceneView_.scene.rootNode.addChildNode(node)
        startNode_ = node
      }
    } else if recognizer.state == .ended || recognizer.state == .cancelled || recognizer.state == .failed {
      startNode_?.removeFromParentNode()
      startNode_ = nil
      lineNode_?.removeFromParentNode()
      lineNode_ = nil
    }
  }

  private func positionOnExistingPlanes() -> SCNVector3? {
    let results = sceneView_.hitTest(view.center, types: .existingPlaneUsingExtent)
    if let result = results.first {
      let transform = result.worldTransform
      let hitPos = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
      return hitPos
    }
    return nil
  }

  private func nodeWithPosition(_ position: SCNVector3) -> SCNNode {
    let sphere = SCNSphere(radius: 0.002)
    sphere.firstMaterial!.diffuse.contents = UIColor.red
    sphere.firstMaterial!.lightingModel = .constant
    sphere.firstMaterial!.isDoubleSided = true
    let node = SCNNode(geometry: sphere)
    node.position = position
    return node
  }

  private func lineNode(from: SCNVector3, to: SCNVector3) -> SCNNode {
    let dirVector = to - from
    let cylinder = SCNCylinder(radius: 0.001, height: CGFloat(dirVector.length()))
    cylinder.radialSegmentCount = 4
    cylinder.firstMaterial?.diffuse.contents = UIColor.green
    cylinder.firstMaterial?.lightingModel = .phong
    let node = SCNNode(geometry: cylinder)
    node.position = SCNVector3.midpointBetween(vector1: from, vector2: to)
    node.eulerAngles = SCNVector3.eulerAngles(vector: dirVector)
    return node
  }

  private func distance(from: SCNVector3?, to: SCNVector3?) -> Float {
    if from == nil || to == nil {
      return 0
    }
    return from!.distance(to: to!)
  }

  private func distanceString(from: SCNVector3?, to: SCNVector3?) -> String {
    let dist = distance(from: from, to: to)
    return String(format: "%.1f cm\n%.1f in", dist * 100.0, dist * 39.37)
  }
}

