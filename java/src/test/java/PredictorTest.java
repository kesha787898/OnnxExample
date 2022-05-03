import ai.catboost.CatBoostError;
import ai.onnxruntime.OrtException;
import org.junit.Assert;
import org.junit.Test;

public class PredictorTest {

    @Test
    public void testPredictorTorch() throws OrtException {

        OnnxPredictor predictor = new OnnxPredictor("//home/kesha787898/IdeaProjects/OnnxExample/python/torch.onnx");
        float[] inp = {(float) 0.4983, (float) 0.4915};
        float[] floats = predictor.predictProba(inp);
        float[] real = {(float) 0.4111};
        Assert.assertArrayEquals(floats, real, (float) 1e-4);
    }

    @Test
    public void testPredictorRF() throws OrtException {

        OnnxPredictor predictor = new OnnxPredictor("//home/kesha787898/IdeaProjects/OnnxExample/python/rf.onnx");
        float[] inp = {(float) 5.1, (float) 3.4, (float) 1.5, (float) 0.2};
        float[] floats = predictor.predictSklearn(inp);
        float[] real = {(float) 9.70724332e-01, (float) 2.92755070e-02, (float) 1.61450810e-07};
        Assert.assertArrayEquals(floats, real, (float) 1e-4);
    }

    @Test
    public void testPredictorCB() throws CatBoostError {

        CatboostPredictor predictor = new CatboostPredictor("//home/kesha787898/IdeaProjects/OnnxExample/python/cb.cbm");
        float[] inp = {(float) 5.1, (float) 3.4, (float) 1.5, (float) 0.2};
        double[] floats = predictor.predictProba(inp);
        double[] real = {9.99380657e-01, 3.43438605e-04, 2.75903968e-04};
        Assert.assertArrayEquals(floats, real, (float) 1e-4);
    }
}
