import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Collections;

public class OnnxPredictor {
    private OrtSession session;
    private static String inputName = "input_1";
    static OrtEnvironment env = OrtEnvironment.getEnvironment();

    public OnnxPredictor(String path) throws OrtException {
        session = env.createSession(path);
    }

    public float[] predictProba(float[] inp) throws OrtException {
        OnnxTensor test = OnnxTensor.createTensor(env, inp);
        OrtSession.Result output = session.run(Collections.singletonMap(inputName, test));
        return (float[]) output.get(0).getValue();
    }

    public float[] predictSklearn(float[] inp) throws OrtException {
        OnnxTensor test = OnnxTensor.createTensor(env, inp);
        OrtSession.Result output = session.run(Collections.singletonMap(inputName, test));
        return ((float[][]) output.get("probabilities").get().getValue())[0];
    }

}