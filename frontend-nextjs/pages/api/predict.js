export default async (req, res) => {
    const response = await fetch('http://flask:5000/predict?text=' + req.query.comment);
    const data = await response.json();
    res.status(200).json(data);
}