import UIKit
import CoreML
import Vision
import ImageIO

class ImageClassificationViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var cameraButton: UIBarButtonItem!
    @IBOutlet weak var classificationLabel: UILabel!

    let model = model_squeezeNet_TSR()

    let labels: [String] = {
        let path = Bundle.main.path(forResource: "signnames", ofType: "csv")!
        let url = URL(fileURLWithPath: path)
        let csv = try! String(contentsOf: url, encoding: .utf8)
        let lines = csv.components(separatedBy: .newlines).dropFirst().filter { !$0.isEmpty }
        return lines.map { $0.components(separatedBy: ",")[1] }
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        print(model.model.modelDescription)
        print(labels)
    }

    func updateClassifications(for image: UIImage) {
        classificationLabel.text = "Classifying..."

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let sSelf = self else { return }
            guard let array = image.preprocess(to: CGSize(width: 32, height: 32)) else {
                DispatchQueue.main.async { sSelf.classificationLabel.text = "Can't convert image to MLMultiArray" }
                return
            }
            do {
                let output = try sSelf.model.prediction(input1: array).output1
                let predictions = output.toArray()
                let max = predictions.max()!
                let maxIndex = predictions.lastIndex(of: max)!
                let maxFormatted = String(format: "%d", Int(max*100.0))
                DispatchQueue.main.async { sSelf.classificationLabel.text = "\(maxFormatted)% \(sSelf.labels[maxIndex])" }
            } catch {
                print("Failed to perform classification.\n\(error.localizedDescription)")
                DispatchQueue.main.async { sSelf.classificationLabel.text = error.localizedDescription }
            }
        }
    }

    @IBAction func takePicture() {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            presentPhotoPicker(sourceType: .photoLibrary)
            return
        }
        
        let photoSourcePicker = UIAlertController()
        let takePhoto = UIAlertAction(title: "Take Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .camera)
        }
        let choosePhoto = UIAlertAction(title: "Choose Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .photoLibrary)
        }
        
        photoSourcePicker.addAction(takePhoto)
        photoSourcePicker.addAction(choosePhoto)
        photoSourcePicker.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        
        present(photoSourcePicker, animated: true)
    }

    func presentPhotoPicker(sourceType: UIImagePickerController.SourceType) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = sourceType
        present(picker, animated: true)
    }
}

extension ImageClassificationViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        let image = info[.originalImage] as! UIImage
        imageView.image = image
        updateClassifications(for: image)
    }

}

// https://github.com/hollance/CoreMLHelpers/issues/5
extension UIImage {

    public func resize(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        self.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()

        return resizedImage
    }
    public func pixelData() -> [UInt8]? {
        let dataSize = size.width * size.height * 4
        var pixelData = [UInt8](repeating: 0, count: Int(dataSize))
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: &pixelData, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 4 * Int(size.width), space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)

        guard let cgImage = self.cgImage else { return nil }
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))

        return pixelData
    }

    func preprocess(to: CGSize) -> MLMultiArray? {


        guard let pixels = self.resize(to: to).pixelData()?.map({ (Double($0) / 255.0 - 0.5) * 2 }) else {
            return nil
        }

        let shape: [NSNumber] = [NSNumber(value: 3), NSNumber(value: Int(to.width)), NSNumber(value: Int(to.height))]
        guard let array = try? MLMultiArray(shape: shape, dataType: .double) else {
            return nil
        }

        let r = pixels.enumerated().filter { $0.offset % 4 == 0 }.map { $0.element }
        let g = pixels.enumerated().filter { $0.offset % 4 == 1 }.map { $0.element }
        let b = pixels.enumerated().filter { $0.offset % 4 == 2 }.map { $0.element }

        let combination = r + g + b
        for (index, element) in combination.enumerated() {
            array[index] = NSNumber(value: element)
        }

        return array
    }
}

extension MLMultiArray {
    func toArray() -> [Double] {
        var a = [Double]()
        for i in 0..<count {
            a.append(self[i].doubleValue)
        }
        return a
    }
}
