// import java.io.*;
//
// class PythonFromJava_JavaModule{
//     public static void main(String args[]){
//         try{
//             int number1 = Integer.parseInt(args[0]);
//             int number2 = Integer.parseInt(args[1]);
//             Process p = Runtime.getRuntime().exec("python PythonFromJava_PythonModule.py " + number1 + " " + number2);
//             BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
//             String ret = new String(in.readLine());
//             System.out.println("value is : " + ret);
//         } catch (Exception e){
//             System.out.println(e);
//         }
//     }
// }